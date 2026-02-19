from __future__ import annotations

import logging
import os
import struct
from typing import List

import pydicom
from pydicom.errors import InvalidDicomError

logger = logging.getLogger(__name__)


def get_file_type(filePath: str) -> str:
    """Get the file type based on the file extension or content.

    Args:
        filePath (str): The path to the file.

    Returns:
        str: The file type (e.g., "Dicom", "Serialized", "txt") or None if unknown.
    """
    # Check for serialized file first (by extension)
    if (
        filePath.endswith(".p")
        or filePath.endswith(".pbz2")
        or filePath.endswith(".pkl")
        or filePath.endswith(".pickle")
    ):
        return "Serialized"

    # Check for txt file
    if filePath.endswith(".txt"):
        return "txt"

    # Check for DICOM file (by extension or by probing content)
    is_dcm_extension = filePath.lower().endswith(".dcm")

    try:
        dcm = pydicom.dcmread(filePath, stop_before_pixels=True, force=is_dcm_extension)
        if dcm is not None:
            return "Dicom"
    except FileNotFoundError:
        logger.debug("File not found while probing for DICOM: %s", filePath)
    except PermissionError:
        logger.warning("Permission denied while probing for DICOM: %s", filePath)
    except IsADirectoryError:
        logger.debug("Encountered directory while probing for DICOM: %s", filePath)
    except InvalidDicomError:
        logger.debug("Not a valid DICOM file: %s", filePath)
    except (OSError, EOFError, struct.error) as exc:
        # Handle truncated/corrupted DICOM files
        if is_dcm_extension:
            logger.warning("Corrupted DICOM file %s: %s", filePath, exc)
            return "Dicom"  # Still classify as DICOM based on extension
        logger.debug("Error while probing for DICOM %s: %s", filePath, exc)
    except Exception as exc:
        logger.warning("Unexpected error while probing for DICOM %s: %s", filePath, exc)
        if is_dcm_extension:
            return "Dicom"  # Still classify as DICOM based on extension

    return None

    return None


def list_all_files(inputPaths: str | List[str], maxDepth: int = -1):
    """
    List all files of compatible data format from given input paths.

    Parameters
    ----------
    inputPaths: str or list of str
        Path or list of paths pointing to the data to be listed.

    maxDepth: int, optional
        Maximum subfolder depth where the function will check for files to be listed.
        Default is -1, which implies recursive search over infinite subfolder depth.

    Returns
    -------
    fileLists: dictionary
        The function returns a dictionary containing lists of data files classified according to their file format (Dicom, MHD).

    """

    fileLists = {"Dicom": [], "MHD": [], "Serialized": [], "txt": []}
    # if inputPaths is a list of path, then iteratively call this function with each path of the list
    if isinstance(inputPaths, list):
        for path in inputPaths:
            lists = list_all_files(path, maxDepth=maxDepth)
            for key in fileLists:
                fileLists[key] += lists[key]

        return fileLists

    # check content of the input path
    if os.path.isdir(inputPaths):
        inputPathContent = sorted(os.listdir(inputPaths))
    else:
        inputPathContent = [inputPaths]
        inputPaths = ""

    for fileName in inputPathContent:
        filePath = os.path.join(inputPaths, fileName)
        # folders
        if os.path.isdir(filePath):
            if maxDepth != 0:
                subfolderFileList = list_all_files(filePath, maxDepth=maxDepth - 1)
                for key in fileLists:
                    fileLists[key] += subfolderFileList[key]

        # files
        elif os.path.isfile(filePath):
            filetype = get_file_type(filePath)
            if filetype is None:
                logging.info("INFO: cannot recognize file format of " + filePath)
            else:
                fileLists[filetype].append(filePath)

    return fileLists
