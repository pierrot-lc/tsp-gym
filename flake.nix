{
  description = "TSP devshell";

  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
      "https://nix-community.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
      "nix-community.cachix.org-1:mB9FSh9qf2dCimDSUo8Zy7bkq5CX+/rkCWyvRCYg3Fs="
    ];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
  let
    system = "x86_64-linux";
    pkgs = import nixpkgs {
      inherit system;
      config = {
        allowUnfree = true;
        cudaSupport = true;
      };
    };

    python-packages = ps: with ps; [
      pip
      setuptools
      virtualenv

      beartype
      einops
      gymnasium
      hydra-core
      jaxtyping
      numpy
      pytest
      torch
      torchinfo
      torchrl
      tqdm
      wandb
    ];

    fhs = pkgs.buildFHSUserEnv {
      name = "tsp-env";
      targetPkgs = pkgs: (with pkgs; [
          (python311.withPackages python-packages)
          cudaPackages.cudatoolkit
          cudaPackages.cudnn
          just
      ]);
    };
  in {
    devShells.${system}.default = fhs.env;
  };
}
