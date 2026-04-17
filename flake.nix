{
  description = "Poetry Python dev shell with CUDA + common native libs";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }:
    let
      systems = [
        "x86_64-linux"
        "aarch64-linux"
      ];

      forAllSystems = f:
        nixpkgs.lib.genAttrs systems (system: f system);
    in
      {
        devShells = forAllSystems (system:
          let
            pkgs = import nixpkgs {
              inherit system;
              config = {
                allowUnfree = true;
                cudaSupport = true;
              };
            };

            python = pkgs.python312;

            runtimeLibs = with pkgs; [
              stdenv.cc.cc.lib
              zlib
              openssl
              libffi
              sqlite
              xz
              bzip2
              readline
              ncurses
              expat
              openblas
              glib
              libGL
              libxkbcommon
              libxcb
              libudev-zero
              alsa-lib
              ffmpeg
              cudaPackages.cudatoolkit
              cudaPackages.cudnn
              wayland
              mesa
              xorg.libX11
              xorg.libXcursor
              xorg.libXi
              xorg.libXrandr
            ];

            fhs = pkgs.buildFHSEnv {
              name = "poetry-open3d-fhs";
              
              targetPkgs = pkgs: (with pkgs; [
                python
                poetry
                pkg-config
                gcc
                gfortran
                gnumake
                cmake
                ninja
                patchelf
                git
              ]) ++ runtimeLibs;

              profile = ''
                export CUDA_PATH="${pkgs.cudaPackages.cudatoolkit}"
                export CUDA_HOME="${pkgs.cudaPackages.cudatoolkit}"
                
                export POETRY_VIRTUALENVS_CREATE="true"
                export POETRY_VIRTUALENVS_IN_PROJECT="true"
                
                export PATH="${python}/bin:$PATH"

                # FORCE X11 TO PREVENT OPEN3D WAYLAND CRASH
                export WAYLAND_DISPLAY=""
                export XDG_SESSION_TYPE="x11"

                if [ -f pyproject.toml ]; then
                  poetry env use ${python}/bin/python >/dev/null 2>&1 || true

                  echo
                  echo "Python: $(which python)"
                  echo "Poetry env: $(poetry env info --path 2>/dev/null || true)"
                  echo "Run once: poetry install"
                  echo "Then:     poetry run python3 main.py"
                  echo
                fi
              '';

              runScript = "bash";
            };
          in
            {
              default = fhs.env;
            });
      };
}
