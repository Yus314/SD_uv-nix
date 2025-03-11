{
  description = "Python venv development template";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    utils.url = "github:numtide/flake-utils";

    pyproject-nix = {
      url = "github:pyproject-nix/pyproject.nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    uv2nix = {
      url = "github:pyproject-nix/uv2nix";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
    pyproject-build-systems = {
      url = "github:pyproject-nix/build-system-pkgs";
      inputs.pyproject-nix.follows = "pyproject-nix";
      inputs.uv2nix.follows = "uv2nix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      utils,
      uv2nix,
      pyproject-nix,
      pyproject-build-systems,
      ...
    }:
    let
      inherit (nixpkgs) lib;
      system = "x86_64-linux";
      workspace = uv2nix.lib.workspace.loadWorkspace { workspaceRoot = ./.; };
      overlay = workspace.mkPyprojectOverlay { sourcePreference = "wheel"; };
      pkgs = import nixpkgs {
        inherit system;
        cudaSupport = true;
      };
      python = pkgs.python310;
      pyprojectOverrides = final: prev: {
        nvidia-cusolver-cu12 = prev.nvidia-cusolver-cu12.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.cudaPackages.cudatoolkit ];
        });
        nvidia-cusparse-cu12 = prev.nvidia-cusparse-cu12.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ [ pkgs.cudaPackages.libnvjitlink.lib ];
        });
        torch = prev.torch.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ [
            pkgs.cudaPackages.cudnn
            pkgs.cudaPackages.nccl
            pkgs.linuxKernel.packages.linux_zen.nvidia_x11
            pkgs.cudaPackages.cudatoolkit
          ];
        });
        torchvision = prev.torchvision.overrideAttrs (old: {
          buildInputs = (old.buildInputs or [ ]) ++ [
            pkgs.cudaPackages.cudnn
            pkgs.cudaPackages.nccl
            pkgs.linuxKernel.packages.linux_zen.nvidia_x11
            pkgs.cudaPackages.cudatoolkit
            (pkgs.libtorch-bin.override { cudaSupport = true; })
          ];
        });
      };
      pythonSet = (pkgs.callPackage pyproject-nix.build.packages { inherit python; }).overrideScope (
        lib.composeManyExtensions [
          pyproject-build-systems.overlays.default
          overlay
          pyprojectOverrides
        ]
      );
    in
    {
      devShells.x86_64-linux.default =
        let
          editableOverlay = workspace.mkEditablePyprojectOverlay {
            root = "./";
            members = [ "hello-world" ];
          };
          editablePythonSet = pythonSet.overrideScope (
            lib.composeManyExtensions [
              editableOverlay
              (final: prev: {
                hello-world = prev.hello-world.overrideAttrs (old: {
                  src = lib.fileset.toSource {
                    root = old.src;
                    fileset = lib.fileset.unions [ (old.src + "/pyproject.toml") ];
                  };
                  nativeBuildInputs =
                    old.nativeBuildInputs
                    ++ final.resolveBuildSystem {
                      editables = [ ];
                      tomli = [ ];
                    };
                });
              })
            ]
          );
          virtualenv = editablePythonSet.mkVirtualEnv "hello-world-dev-env" workspace.deps.all;
        in

        pkgs.mkShell {
          buildInputs = with pkgs; [
            git
            gitRepo
            gnupg
            autoconf
            curl
            procps
            gnumake
            util-linux
            m4
            gperf
            unzip
            libGLU
            libGL
            xorg.libXi
            xorg.libXmu
            freeglut
            xorg.libXext
            xorg.libX11
            xorg.libXv
            xorg.libXrandr
            zlib
            ncurses5
            binutils
            stdenv.cc.cc.lib
            libGL
            glib
            cudaPackages.cudatoolkit
            graphviz
            linuxPackages.nvidia_x11
          ];
          packages = [
            virtualenv
            pkgs.uv
            pkgs.ollama
          ];
          env = {
            LD_LIBRARY_PATH = "${
              with pkgs;
              lib.makeLibraryPath [
                zlib
                stdenv.cc.cc.lib
                libGL
                glib
              ]
            }:run/opengl-driver/lib";
            UV_NO_SYNC = "1";
            UV_PYTHON_DOWNLOADS = "never";
            UV_PYTHON = "${virtualenv}/bin/python";
          };

          shellHook = ''
                                                                        	    unset PYTHONPATH
                                                				    export CUDA_PATH=${pkgs.cudaPackages.cudatoolkit}
                        export EXTRA_CCFLAGS="-I/usr/include"
            	    export EXTRA_LDFLAGS="-L/lib -L${pkgs.linuxPackages.nvidia_x11}/lib"
                                                            			export REPO_ROOT=$(git rev-parse --show-toplevel)
                                    			
                                                            	  '';
        };
    };
}
