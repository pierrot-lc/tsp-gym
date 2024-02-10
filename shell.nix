{
  pkgs ? import <nixpkgs> {
    config = {
      allowUnfree = true;
      CudaSupport = true;
    };
  }
}:
let
  torchPackage = pkgs.python311Packages.torch-bin;
  python-packages = ps: with ps; [
    pip
    setuptools
    virtualenv

    einops
    gymnasium
    hydra-core
    numpy
    pytest
    torchPackage
    ( torchinfo.override { torch = torchPackage; } )
    ( torchrl.override { torch = torchPackage; } )
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
