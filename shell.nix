{
  pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      CudaSupport = true;
    };
  }
}:
let
  python-packages = ps: with ps; [
    pip
    setuptools
    virtualenv

    einops
    gymnasium
    hydra-core
    numpy
    pytest
    torchWithCuda
    torchinfo
    torchrl
    tqdm
    wandb
  ];
in
  (pkgs.buildFHSUserEnv {
    name = "pytorch";
    targetPkgs = pkgs: (with pkgs; [
      (python311.withPackages python-packages)
      cudaPackages.cudatoolkit
      cudaPackages.cudnn
      just
    ]);
    runScript = "bash";
  }).env
