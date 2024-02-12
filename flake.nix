{
  description = "TSP devshell";

  nixConfig = {
    extra-substituters = [
      "https://cuda-maintainers.cachix.org"
    ];
    extra-trusted-public-keys = [
      "cuda-maintainers.cachix.org-1:0dq3bujKpuEPMCX6U4WylrUDZ9JyUG0VpVZa7CNfq5E="
    ];
  };

  inputs = {
    nixpkgs.url = "nixpkgs/nixos-23.11";
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

      # Override all torch packages for `torch-bin`.
      # This is important since the other torch derivatives
      # need a default torch package and assume it's `torch`.
      # Taken from: https://discourse.nixos.org/t/overriding-torch-with-torch-bin-for-all-packages/37086/2.
      overlays = [
        (final: prev: {
          pythonPackagesExtensions = [
            (py-final: py-prev: {
              torch = py-final.torch-bin;
            })
          ];
        })
      ];
    };

    python-packages = ps: with ps; [
      pip
      setuptools
      virtualenv

      einops
      gymnasium
      hydra-core
      numpy
      pytest
      torch-bin
      # torchinfo
      # torchrl
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
