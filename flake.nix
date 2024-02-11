{
  description = "TSP devshell";

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
  in {
    devShells.${system} = {
      default = pkgs.mkShell {
        name = "tsp";
        buildInputs = with pkgs; [
          (python311.withPackages python-packages)
          cudaPackages.cudatoolkit
          cudaPackages.cudnn
          just
        ];
      };
    };
  };
}
