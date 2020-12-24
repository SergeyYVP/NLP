import numpy as np
import pickle
# from bs4 import BeautifulSoup as bs
import urllib
from urllib.request import urlopen, Request


def save_dict(dct_str, fnc, *args):
    try:
        with open("data/" + dct_str + ".pkl", "rb") as f:
            dct = pickle.load(f)
    except:
        dct = fnc(*args)
        with open("data/" + dct_str + ".pkl", "wb") as f:
            pickle.dump(dct, f)
    return dct


def save_obj(name, obj):
    with open(name, "wb") as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name, "rb") as f:
        return pickle.load(f)


def mcc_description(mcc_list):
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 6.1) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/41.0.2228.0 Safari/537.3"
    }
    reg_url = "https://mcc-codes.ru/code"
    req = Request(url=reg_url, headers=headers)
    check = bs(urlopen(req).read())
    list_of_mcc_tags = check.find_all("tr")[1:]
    mcc_dict = dict(
        map(lambda x: (x.text.split("\n")[1], x.text.split("\n")[2]), list_of_mcc_tags)
    )
    return mcc_dict


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024 ** 2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                # if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                #     df[col] = df[col].astype(np.float16)
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024 ** 2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / start_mem
            )
        )
    return df


def tokenize_en(sentence, en):
    return [tok.text for tok in en.tokenizer(sentence)]

def tokenize_de(sentence, de):
    return [tok.text for tok in de.tokenizer(sentence)]
