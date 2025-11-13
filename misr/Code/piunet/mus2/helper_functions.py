from os.path import exists, join, isfile
from os import makedirs, listdir


def get_all_files(path):
    """Extracts files paths from a given directory.

    Parameters
    ----------
    path : str
         Path to the directory containing files.

    Returns
    -------
    files_list : list
        List of full files paths.

   """
    return [join(path, f) for f in listdir(path) if isfile(join(path, f))]


def create_directory(path):
    if not exists(path):
        makedirs(path)
