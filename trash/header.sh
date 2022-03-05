#!/usr/bin/env bash
#-----------------------------------------------------------------------------#
# Basic functions
#-----------------------------------------------------------------------------#
# Echo error message
echoerr() {
  echo "$*" 1>&2
}

# Raise error
raise() {
  local code message
  if [ $# -gt 1 ] && [ -z "${1//[0-9]/}" ]; then
    code=$1
    message=${*:2}
  else
    code=1
    message=$*
  fi
  # shellcheck disable=2154
  if [ -n "$usage" ]; then
    echo "Usage: $usage" >&2
  fi
  echo "Error: $message" >&2
  exit $code
}

# Check if destination file is *newer* than source files
newer() {
  local src file
  dest=$1
  src=("${@:2}")
  if [ ${#src[@]} -eq 0 ]; then
    echoerr "Skipping (no source files found)."
    return 0
  fi

  # Test existence and format of source and destination files
  idest=${dest%%.*}.nc
  [ "$dest" != "$idest" ] && [ -r "$idest" ] && {
    rm "$idest"
    echoerr "Warning: Removing ${idest} (has no day string)."
  }
  for file in "${src[@]}"; do
    [ -r "$file" ] || {  # TODO: test with ncdump? nah...
      echoerr "Warning: Skipping (source file ${file} is unreadable)."
      return 0  # do not re-process since source is unavailable!
    }
  done
  [ -r "$dest" ] || {
    echoerr "Running (${dest##*/} not found)."
    return 1  # re-process since destination is unavailable!
  }

  # Test modification dates and validity of source files
  for file in "${src[@]}"; do
    [ "$file" -nt "$dest" ] && {
      echoerr "Running (${dest##*/} older than source file(s))."
      return 1  # re-process since destination is *not* newer
    }
  done
  ncdump -h "$dest" &>/dev/null || {
    echoerr "Running (${dest##*/} appears corrupt)."
    return 1  # re-process since destination is corrupt
  }
  ntime=$(ncdump -h "$dest" | grep UNLIMITED | tr -dc 0-9)  # format should be 'time = UMLIMITED; // (0 currently)'
  [ -n "$ntime" ] && [ "$ntime" -eq 0 ] && {
    echoerr "Running (${dest##*/} has zero-length time dim)."
    return 1  # re-process since destination is invalid (prevoius computation error)
  }
  if $overwrite; then
    echoerr "Overwriting up-to-date file."
    return 1  # re-process since overwrite requested
  else
    echoerr "Skipping (${dest##*/} exists and is new)."
    return 0  # do not re-process since recently done
  fi
}
