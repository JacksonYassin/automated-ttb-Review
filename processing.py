import easyocr
from parsing import *
from scanning import *


##############
#
# Driver code to process and verify labels
#
##############


def process_label(image:str, reader:easyocr.Reader, app_info:list):
    """
    Driver function to process an image and verify all components are present
    """
    # scan image
    tess, easy = ocr_scan(image, reader)
    # make fusion list
    similar_list = make_fusion_list(tess, easy)
    # fuse the lists
    fused_set = fuse_lists(similar_list)
    # find elements with example data
    found = find_elements(app_info, fused_set, tess, easy)
    # test alcohol and net content
    valid_alc_net_content = verify_locations(found, fused_set, tess, easy)
    # test government warning label
    valid_gov_warning = verify_government_warning(fused_set, tess, easy)
    # create final output list
    final_output = [
        (True, found[i][0][1]) if found[i] and found[i] != "phrase not found" else (False, None)
        for i in range(len(app_info))
        ] + valid_alc_net_content + [valid_gov_warning]

    return final_output


def evaluate_label_results(final_output:list, labels:list=None):
    """
    Returns "pass" if all elements are found
    Returns "fail" and list of not-present elements otherwise 
    """

    failed = []

    for idx, (status, payload) in enumerate(final_output):
        if not status and idx != 2:
            if labels and idx < len(labels):
                failed.append(labels[idx])
            else:
                failed.append({
                    "index": idx,
                    "payload": payload
                })

    if not failed:
        return 0

    return 1, failed

