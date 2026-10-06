from pathlib import Path

root = Path(r'C:\Users\John Arcilla\Documents\pawsadaan-ai-service-v2')
files = {
    'Dockerfile': '''FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \\
    PIP_NO_CACHE_DIR=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
WORKDIR /app
COPY requirements.txt .
RUN python -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu \\
    && python -m pip install -r requirements.txt
COPY main.py model_loader.py ./
COPY weights/ ./weights/
RUN python -c "import hashlib; from pathlib import Path; p=Path('weights/dinov2-vits14-arcface-pawsadaan-v1.pt'); assert hashlib.sha256(p.read_bytes()).hexdigest() == '3bde378a32d8499f82f9bf6e179810f7c4c92f7a233e0254a612defc5e7bf495'"
EXPOSE 10000
CMD ["sh", "-c", "exec uvicorn main:app --host 0.0.0.0 --port ${PORT:-10000} --workers 1 --limit-concurrency 4"]
''',
    '.dockerignore': '''.git
.env
.venv
venv
__pycache__
**/__pycache__
*.pyc
tests
''',
    'render.yaml': '''services:
  - type: web
    name: pawsadaan-ai-service-v2
    runtime: docker
    plan: free
    region: singapore
    branch: main
    dockerfilePath: ./Dockerfile
    healthCheckPath: /health
    envVars:
      - key: AI_SERVICE_TOKEN
        generateValue: true
      - key: MODEL_WEIGHTS_PATH
        value: ./weights/dinov2-vits14-arcface-pawsadaan-v1.pt
      - key: OMP_NUM_THREADS
        value: "1"
      - key: MKL_NUM_THREADS
        value: "1"
''',
}
for name, content in files.items():
    path = root / name
    if path.exists():
        raise RuntimeError(f'Refusing to overwrite {path}')
    path.write_text(content, encoding='utf-8', newline='\n')

loader = root / 'model_loader.py'
text = loader.read_text(encoding='utf-8')
text = text.replace('torch.load(path,map_location="cpu",weights_only=True)',
                    'torch.load(path,map_location="cpu",weights_only=True,mmap=True)')
text = text.replace('    model = EmbeddingNet().eval().requires_grad_(False)\n    model.load_state_dict(bundle["state_dict"],strict=True)',
                    '    # Avoid allocating a second full copy of weights during startup.\n'
                    '    with torch.device("meta"):\n'
                    '        model = EmbeddingNet()\n'
                    '    model.load_state_dict(bundle["state_dict"],strict=True,assign=True)\n'
                    '    model.eval().requires_grad_(False)')
loader.write_text(text, encoding='utf-8', newline='\n')

ignore = root / '.gitignore'
ignore.write_text(ignore.read_text() + '\n# Approved deployment artifact (other checkpoints stay ignored).\n!weights/dinov2-vits14-arcface-pawsadaan-v1.pt\n', encoding='utf-8', newline='\n')

readme = root / 'README.md'
text = readme.read_text(encoding='utf-8').replace(
    'Place `dinov2-vits14-arcface-pawsadaan-v1.pt` from the training export in\n'
    '`weights/` before starting. The local prepared copy already includes it.\n'
    'Weights are excluded from Git; transfer the artifact separately to each host.',
    'The approved `dinov2-vits14-arcface-pawsadaan-v1.pt` export is included in\n'
    '`weights/` so a fresh clone or Render build has everything needed for inference.\n'
    'Other training checkpoints remain excluded from Git.')
text += '''
## Render (free-tier testing)

1. In Render choose **New > Blueprint** and connect this repository, branch `main`.
2. Review `render.yaml`: one **Free** Docker web service in Singapore, no disk or database.
3. Deploy. The Docker build installs CPU-only PyTorch and includes the verified weights.
4. Wait for **Live**, then open `https://<your-service>.onrender.com/health`.
5. In the service's **Environment**, reveal/copy the generated `AI_SERVICE_TOKEN`
   for your backend or API client. Keep it secret; do not put it in Git or browser frontend code.

If creating a Web Service manually, choose Docker, Free, health check `/health`,
and add a long random `AI_SERVICE_TOKEN`. Leave Docker Command empty; the image
binds to Render's `$PORT` on `0.0.0.0` and runs one worker.

Open `/docs`, expand `POST /embed`, click **Try it out**, fill
`x-pawsadaan-ai-token`, select a cat photo, and execute. Expect 512 embedding
values, the model version above and threshold `0.5165491104125977`.
An incorrect token must return 401.

The free instance has 512 MB RAM and sleeps after 15 idle minutes. Initial requests
after sleep can take about a minute. CPU-only packages and memory-mapped model
loading reduce memory use, but successful inference within the live memory limit
must be verified. If logs report out-of-memory, stop and reassess; do not enable a
paid instance automatically. Free instance hours are shared across the workspace.

For backend testing, set `AI_SERVICE_URL` to the HTTPS Render URL and configure
the matching token in the backend's existing service-token setting, plus the V2
model version, dimensions, and threshold listed above. Refresh stored embeddings
as a separate coordinated step. A live `/health` alone does not verify `/embed`.
'''
readme.write_text(text, encoding='utf-8', newline='\n')
print('Prepared Dockerfile, Render Blueprint, model loading and deployment instructions.')
