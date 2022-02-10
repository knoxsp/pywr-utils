# General Pywr Utilities

A set of general command-line utilities to run, inspect and manipulate pywr models. 

```
Usage: pywr-utils [OPTIONS] COMMAND [ARGS]...

Options:
  --help  Show this message and exit.

Commands:
  remove-orphan-parameters  Remove any parameters from a model which are...
  run-file                  Run pywr on the specified file

```

## run-file
Runs a pywr model with the specified JSON file. 
```<bash>
pywr-utils run-file /tmp/my_pywr_model.json
```

## remove-orphan-parameters
Creates a new copy of a specified JSON file, but removes any parameters from the model which are not used.
This was created after merging multiple model files resulted in too many parameters, which caused the model to not run. Removing the unused parameters fixed the issue.

```
pywr-utils remove-orphan-parameters /tmp/my_pywr_model.json
```
