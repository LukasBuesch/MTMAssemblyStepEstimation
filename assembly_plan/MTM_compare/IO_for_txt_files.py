#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from os import path
import time as t


def read_from_txt(filename, folder, get_settings=False):
    """
    read data from file
    :param get_settings:
    :param folder: Name of folder where file is stored
    :param filename: name of file to read
    :return:
    """

    '''create rel_path to file'''
    file_rel_path = get_rel_path(filename, folder)

    ''' begin writing '''

    with open(file_rel_path, 'r') as file:
        load_description = True
        load_settings = False
        load_list = False
        list_append = []
        list_settings = []
        for i, line in enumerate(file):

            if line == '### begin settings\n':
                # print('settings found:')
                load_description = False
                load_settings = True
                load_list = False
                continue

            elif line == '### begin data\n':
                load_description = False
                load_settings = False
                load_list = True
                # print('data found:')
                continue

            elif line == '\n':
                continue

            if load_settings and not load_list:
                list_settings.append(line[:-1].split(' '))

            if load_list and not load_settings:
                list_append.append(line[:].split('\n')[0])

        data_settings = np.asarray(list_settings)
        data = list_append

        file.close()

    if get_settings:
        return data, data_settings

    else:
        return data


def write_to_txt(filename, data, file_description='', file_settings='', append=0, folder=None):
    """
    write data into text file

    :param file_settings:
    :param append: 0 for create new file ; 1 for append to file
    :param folder: folder to store file in (with '/' at end)
    :param data: Data to write in .txt file as array -> every row in data gets one row in
    txt file (e.g: data = np.asarray([[123, 2000, 1], [...], [...]]))

    :param filename: without ending '.txt'
    :param file_description: e.g.: = ('Measurement simulation file\n')
    :return:
    """

    if append == 0:
        entry = 'w'

    else:
        entry = 'a'

    '''create rel_path to file'''
    file_rel_path = get_rel_path(filename, folder)

    with open(file_rel_path, entry) as file:

        # write header to file if append == 0
        if append == 0:
            file.write(file_description + '\n')

            file.write('### begin settings\n')

            file.write(file_settings + '\n')

            ''' begin to write data '''

            file.write('### begin data\n')

        for i in range(len(data)):

            for itx in range(len(data[0])):
                file.write(str(data[i][itx]) + ' ')

            file.write('\n')

        file.close()

    return True


def delete_line_in_file(personal_id, filename, folder):
    """

    :param personal_id:
    :param filename:
    :param folder:
    :return:
    """

    '''create rel_path to file'''
    file_rel_path = get_rel_path(filename, folder)

    year = ' ' + str(t.localtime()[0])

    with open(file_rel_path, 'r') as r_file:
        with open(get_rel_path('name_deletions', 'names'), 'a') as junk_stor:

            lines = []
            for i, line in enumerate(r_file):
                try:
                    if line.split(' ')[0] != personal_id:
                        lines.append(line)
                    elif line.split(' ')[0] == personal_id:
                        line = '\n' + line[:-2] + year
                        junk_stor.write(line)

                except:
                    lines.append(line)
            junk_stor.close()
            r_file.close()

    with open(file_rel_path, 'w') as w_file:
        for line in lines:
            w_file.write(line)
        w_file.close()
    return 0


def get_rel_path(filename, folder):
    """create rel_path to file"""
    if folder is not None:
        if folder == 'path':
            file_rel_path = filename

        else:
            file_rel_path = path.relpath(folder + '/' + filename + '.txt')

    else:
        file_rel_path = filename + '.txt'
    return file_rel_path


if __name__ == '__main__':
    pass
