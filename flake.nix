{
  description = "Next Agent Development Flake";

  inputs = {
    pyproject.url = "github:ret2pop/pyproject_tools";
  };
  outputs = { self, pyproject }:
    {
      devShells = pyproject.devShells;
    };
}
