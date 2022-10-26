# remove conda environment for current directory
function rmcenv()
{
  if [[ -f ".cenv" ]]; then

    cenv_name="$(<.cenv)"

    # detect if we need to switch conda environment first
    if [[ -n "$CONDA_DEFAULT_ENV" ]]; then
        current_cenv="$(basename $CONDA_DEFAULT_ENV)"
        if [[ "$current_cenv" = "$cenv_name" ]]; then
            _default_cenv
        fi
    fi

    conda env remove --name "$cenv_name"
    rm ".cenv"
  else
    printf "No .cenv file in the current directory!\n"
  fi
}


# helper function to create a conda environment for the current directory
function mkcenv()
{
  if [[ -f ".cenv" ]]; then
    printf ".cenv file already exists. If this is a mistake use the rmcenv command\n"
  else
    cenv_name="$(basename $PWD)"
    conda create --name "$cenv_name" $@
    conda run -n "$cenv_name" poetry install
    printf "$cenv_name\n" > ".cenv"
    chmod 600 .cenv
  fi
}

mkcenv python=3.9.13 -y
