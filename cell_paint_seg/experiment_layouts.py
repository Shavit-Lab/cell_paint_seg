

def id_from_name_dataset(name):
    id = name[:-7]
    return id 

def condition_from_id(id):
    row_to_roid = {"B":0, "C":1, "D":2, "E":3, "F":4, "G":5}

    well = id.split("_")[1]
    row = well[1]
    row = row_to_roid[row]
    col = int(well[2:])
    col -= 2

    conditions = [[1,1,2,2,5,1,1,4,5,5],
                  [3,3,4,4,5,2,2,4,3,3],
                  [5,2,2,1,1,2,2,3,3,5],
                  [3,3,4,4,5,5,4,4,1,1],
                  [4,4,1,1,3,2,2,1,3,3],
                  [5,5,2,2,3,5,5,1,4,4]]
    
    return conditions[row][col]

def line_from_id(id):
    row_to_roid = {"B":0, "C":1, "D":2, "E":3, "F":4, "G":5}

    e = int(id[1])
    well = id.split("_")[1]
    row = well[1]
    row = row_to_roid[row]
    col = int(well[2:])

    # Control: AE8, ZKZ, XH7, ADK, NK3
    # C9 ALS: RFT, TJV, DG9, EGM, LJX
    # SOD1 ALS: BFU, KRC, RJV, ZLM, AF6
    if row < 2:
        if col < 7:
            e_to_line = {1:"RFTiALS", 2:"AE8iCTR", 3:"ADKiCTR", 4:"EGMiALS", 5: "ZLMiALS", 6: "ZLMiALS"}
        else:
            e_to_line = {1:"AE8iCTR", 2:"BFUiALS", 3:"ZLMiALS", 4:"XH7iCTR", 5: "RFTiALS", 6: "RFTiALS"}
    elif row < 4:
        if col < 7:
            e_to_line = {1:"ZKZiCTR", 2:"KRCiALS", 3:"BFUiALS", 4:"ADKiCTR", 5: "EGMiALS", 6: "EGMiALS"}
        else:
            e_to_line = {1:"TJViALS", 2:"XH7iCTR", 3:"NK3iCTR", 4:"LJXiALS", 5: "RJViALS", 6: "RJViALS"}
    else:
        if col < 7:
            e_to_line = {1:"DG9iALS", 2:"ZKZiCTR", 3:"ZKZiCTR", 4:"NK3iCTR", 5: "BFUiALS", 6: "BFUiALS"}
        else:
            e_to_line = {1:"XH7iCTR", 2:"RJViALS", 3:"AFGiALS", 4:"AE8iCTR", 5: "DG9iALS", 6: "DG9iALS"}
    
    
    return e_to_line[e]

def line_to_class(line):
    if "CTR" in line:
        return 0
    # C9
    elif line in ["RFTiALS", "EGMiALS", "TJViALS", "LJXiALS", "DG9iALS"]:
        return 1
    else:
        return 2
    
def condition_num_to_name(num, inverse = False):
    cond_to_cond = {1: "KPT", 2:"H2O2", 3:"Tunicamycin", 4:"Autophagy", 5:"DMSO"}

    # flip dictionary so values are keys
    if inverse:
        cond_to_cond = {v: k for k, v in cond_to_cond.items()}
        return cond_to_cond[num]
    else:
        return cond_to_cond[num]

def channel_to_name(channel, inverse = False):
    channels = ["ER", "DNA", "Mito", "Actin", "RNA", "Golgi/membrane"]
    if inverse:
        channels = {v: k for k, v in enumerate(channels)}
        return channels[channel]
    else:
        return channels[channel]