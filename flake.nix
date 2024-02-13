{
  inputs = {
    # bisect 126f49a01de5b7e35a43fd43f891ecf6d3a51459 for breakage with newer verison
    nixpkgs.url = "github:NixOS/nixpkgs/e32eb6818377a7d3ed2d0815418e6cb8427819e7";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
  };

  outputs =
    {
      nixpkgs,
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
          pkgs = import nixpkgs {
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
                env = with pkgs; {
                  LD_LIBRARY_PATH = "${stdenv.cc.cc.lib}/lib ";
                };

                # https://devenv.sh/reference/options/
                packages = with pkgs; [
                  oxipng
                  realesrgan-ncnn-vulkan
                ];

                dotenv.disableHint = true;

                # python
                languages.python = {
                  enable = true;
                  package = pkgs.python3.withPackages (
                    ps: with ps; [
                      (mmcv.overridePythonAttrs (
                        old: rec {
                          # needed for mmpose < 1.0
                          # mmpose 1.0+ was a major change that broke imports and loading model data
                          version = "1.7.0";
                          src = pkgs.fetchFromGitHub {
                            owner = "open-mmlab";
                            repo = "mmcv";
                            rev = "v${version}";
                            hash = "sha256-EVu6D6rTeebTKFCMNIbgQpvBS52TKk3vy2ReReJ9VQE=";
                          };

                          doCheck = false;
                        }
                      ))
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
