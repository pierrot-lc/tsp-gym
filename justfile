tests:
  python3 -m pytest --import-mode importlib .

dev:
  nix develop --verbose --accept-flake-config

cuda:
  python3 -c "import torch; print(torch.cuda.is_available())"
