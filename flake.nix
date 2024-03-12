{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
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
                  (callPackage ./nix/realcugan-ncnn-vulkan { })
                ];

                dotenv.disableHint = true;

                # python
                languages.python = {
                  enable = true;
                  # provide hard to compile packages to pip
                  package = pkgs-mmcv.python3.withPackages (
                    ps:
                    let
                      mmcv-patched = ps.callPackage ./nix/mmcv { };
                    in
                    with ps;
                    [
                      mmcv-patched
                      (ps.callPackage ./nix/mmdet { mmcv = mmcv-patched; })
                      (ps.callPackage ./nix/mmpose {
                        mmcv = mmcv-patched;
                        xtcocotools = ps.callPackage ./nix/xtcocotools { };
                      })
                      numpy
                      pillow
                      flake8
                      black
                    ]
                  );
                };
              }
            ];
          };
        }
      );

      packages = forEachSystem (
        system:
        let
          pkgs = import nixpkgs { inherit system; };
        in
        {
          realcugan-ncnn-vulkan = pkgs.callPackage ./nix/realcugan-ncnn-vulkan { };
        }
      );
    };
}
