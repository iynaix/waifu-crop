{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/5103bfc5bfe69a54a6dca9fe473d063b058fc4a9";
    # mmcv fails to build because of newer torch version
    nixpkgs-mmcv.url = "github:NixOS/nixpkgs/5a623156afb531ba64c69363776bb2b2fe55e46b";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
  };

  outputs =
    {
      nixpkgs,
      nixpkgs-mmcv,
      devenv,
      systems,
      ...
    }@inputs:
    let
      forEachSystem = nixpkgs.lib.genAttrs (import systems);
    in
    {
      devShells = forEachSystem (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
          pkgs-mmcv = import nixpkgs-mmcv {
            inherit system;
            # opencv4 override is done in an overlay as it is a dependency
            # for multiple other python libraries and will have conflicts
            # otherwise
            overlays = [
              (final: prev: {
                # need gtk support for opencv to show the preview window
                opencv4 = prev.opencv4.override { enableGtk3 = true; };
              })
            ];
          };
        in
        {
          default = devenv.lib.mkShell {
            inherit inputs pkgs;
            modules = [
              {
                # https://devenv.sh/reference/options/
                packages = with pkgs; [
                  oxipng
                  realesrgan-ncnn-vulkan
                ];

                dotenv.disableHint = true;

                # python
                languages.python = {
                  enable = true;
                  # provide hard to compile packages to pip
                  package = pkgs-mmcv.python3.withPackages (
                    ps: with ps; [
                      # needed for mmpose < 1.0
                      # mmpose 1.0+ was a major change that broke imports and loading model data
                      (mmcv.overridePythonAttrs (old: rec {
                        version = "1.7.0";
                        src = pkgs.fetchFromGitHub {
                          owner = "open-mmlab";
                          repo = "mmcv";
                          rev = "v${version}";
                          hash = "sha256-EVu6D6rTeebTKFCMNIbgQpvBS52TKk3vy2ReReJ9VQE=";
                        };

                        doCheck = false;
                      }))
                      numpy
                      pillow
                      flake8
                      black
                      torch
                      torchvision
                    ]
                  );
                  venv.enable = true;
                };
              }
            ];
          };
        }
      );
    };
}
