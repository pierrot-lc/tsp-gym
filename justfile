tests:
  python3 -m pytest --import-mode importlib .

cuda:
  python3 -c "import torch; print(torch.cuda.is_available())"
