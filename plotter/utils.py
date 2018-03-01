#!/bin/python3
import csv
import pandas as pd
import numpy as np
import functools

# data management tools


# def consolidate_dict_data(dict_data: dict, consolidate_argx, consolidate_argy, consolidate_argz,
#                           argx_name="arg_x", argy_name="arg_y", filters=None):
#     consolidate_dict = dict()
#     for keys, values in dict_explore(dict_data):
#         arg_x = keys[consolidate_argx]
#         arg_y = keys[consolidate_argy]
#         arg_z = keys[consolidate_argz]
#         arg_xy = (arg_x, arg_y)
#         if filters is None or are_keys_on_filters(keys, filters):
#             if arg_z not in consolidate_dict:
#                 consolidate_dict[arg_z] = dict()
#             if arg_xy not in consolidate_dict[arg_z]:
#                 consolidate_dict[arg_z][arg_xy] = list()
#             consolidate_dict[arg_z][arg_xy].append(values)
#
#     return generate_consolidate_dict(consolidate_dict, argx_name, argy_name)


def consolidate_array_data(array_data: pd.DataFrame, consolidate_argx, consolidate_argy, consolidate_argz,
                           argx_name="arg_x", argy_name="arg_y", filters=None):
    arg_x_id = array_data.columns.get_loc(consolidate_argx)
    arg_y_id = array_data.columns.get_loc(consolidate_argy)
    arg_z_id = array_data.columns.get_loc(consolidate_argz)
    parsed_filters = dict()
    if filters is not None:
        for key, value in filters:
            i = array_data.columns.get_loc(key)
            parsed_filters[i] = value
    # filtering_function = functools.partial(are_keys_on_filters, filters=parsed_filters)
    consolidate_dict = dict()
    for row in array_data.itertuples():  # filter(filtering_function, array_data.itertuples()):
        arg_x = row[arg_x_id]
        arg_y = row[arg_y_id]
        arg_z = row[arg_z_id]
        arg_xy = (arg_x, arg_y)
        if filters is not None or are_keys_on_filters(row, parsed_filters):
            if arg_z not in consolidate_dict:
                consolidate_dict[arg_z] = dict()
            if arg_xy not in consolidate_dict[arg_z]:
                consolidate_dict[arg_z][arg_xy] = list()
            consolidate_dict[arg_z][arg_xy].append(row[2])

    return generate_consolidate_dict(consolidate_dict, argx_name, argy_name)


def generate_consolidate_dict(filtered_dict, argx_name, argy_name):
    return_dict = dict()
    for arg_z, d in filtered_dict.items():
        data = []
        for k, l in d.items():
            a = np.array(l)
            mean = np.mean(a)
            std = np.std(a)
            median = np.median(a)
            a_min = np.min(a)
            a_max = np.max(a)
            data.append([k[0], k[1], mean, median, std, a_min, a_max])
        df = pd.DataFrame(data, columns=[argx_name, argy_name, "mean", "median", "std", "min", "max"])
        return_dict[arg_z] = df
    return return_dict


def are_keys_on_filters(keys, filters: dict):
    for key_index, key_value in filters.items():
        try:
            if keys[key_index] not in key_value:
                return False
        except TypeError:
            if keys[key_index] != key_value:
                return False
    return True


def dict_explore(dict_data, keys=None):
    if keys is None:
        keys = []
    if type(dict_data) == dict:
        for k, v in dict_data.items():
            keys.append(k)
            yield from dict_explore(v, keys)
            keys.pop()
    else:
        yield keys, dict_data


def to_seaborn_dataframe(consolidate_dict, wanted_value='median', value_name='median', consolidate_z_name='argz'):
    df = pd.DataFrame()
    for k, v in consolidate_dict.items():
        tmp = pd.DataFrame(data=v.get([v.columns[0], v.columns[1], wanted_value]),
                           columns=[v.columns[0], v.columns[1], value_name])
        tmp = tmp.assign(argz=[k] * len(tmp))
        tmp.rename(columns={'argz': consolidate_z_name}, inplace=True)
        df = df.append(tmp)
        df.columns = tmp.columns
    return df


if __name__ == '__main__':
    df = pd.read_csv("StarcraftTestErrors.csv")
    dd = consolidate_array_data(df, "epoch", "error", "Opt_name", filters=[("Opt_name", ["SA"])])
