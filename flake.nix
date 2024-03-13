{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    systems.url = "github:nix-systems/default";
    devenv.url = "github:cachix/devenv";
    anime-face-detector.url = "github:iynaix/anime-face-detector";
  };

  outputs =
    {
      nixpkgs,
      devenv,
      anime-face-detector,
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
        in
        {
          default = devenv.lib.mkShell {
            inherit inputs pkgs;
            modules = [
              {
                # https://devenv.sh/reference/options/
                packages = with pkgs; [
                  oxipng
                  anime-face-detector.packages.${system}.anime-face-detector
                  (callPackage ./nix/realcugan-ncnn-vulkan { })
                ];

                dotenv.disableHint = true;

                # python
                languages.python = {
                  enable = true;
                  # provide hard to compile packages to pip
                  package = pkgs.python3.withPackages (
                    ps: with ps; [
                      numpy
                      pillow
                      flake8
                      black
                      (opencv4.override { enableGtk3 = true; })
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
