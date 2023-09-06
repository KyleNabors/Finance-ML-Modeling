import datetime
import numpy as np
import json


def myconverter(o):

    if isinstance(o, datetime.datetime):
        return o.__str__()
    if isinstance(o, np.bool_):
        return o.__str__()


def writeToJSONFile(filepath, data):
    with open(filepath, 'w') as file:
        json.dump(data, file, default=myconverter, indent=4)


ports = {
    "data_extraction": 5000
}

PREFIX = "/api"

import os
from PyPDF2 import PdfFileReader, PdfFileWriter


def splitting(upload_folder, split_folder):
    '''Do collect PDF files, split pages and save them
    '''

    entries = os.listdir(upload_folder)
    path = os.path.abspath(split_folder)

    for entry in entries:

        uploaded_file = os.path.join(upload_folder, entry)
        output_file_folder = os.path.join(path, entry)

        if not os.path.isdir(output_file_folder):
            os.mkdir(output_file_folder)

            pdf = PdfFileReader(uploaded_file, strict=False)
            for page in range(pdf.getNumPages()):
                pdf_writer = PdfFileWriter()
                pdf_writer.addPage(pdf.getPage(page))
                output_filename = \
                    os.path.join(output_file_folder, f'{page+1}.pdf')
                with open(output_filename, 'wb') as out:
                    pdf_writer.write(out)

import os
from io import StringIO
import re
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import TextConverter
from pdfminer.layout import LAParams
from pdfminer.pdfpage import PDFPage


def pdf_to_text(path):
    '''Extract text from pdf documents
    '''

    manager = PDFResourceManager()
    retstr = StringIO()
    layout = LAParams(all_texts=False, detect_vertical=True)
    device = TextConverter(manager, retstr, laparams=layout)
    interpreter = PDFPageInterpreter(manager, device)
    with open(path, 'rb') as filepath:
        for page in PDFPage.get_pages(filepath, check_extractable=True):
            interpreter.process_page(page)
    text = retstr.getvalue()
    device.close()
    retstr.close()
    return text


def extraction(split_path, text_path):
    '''Extract and save text files to output dir
    '''

    # entries names
    entries = os.listdir(split_path)

    # repeat the process for each entry
    for entry in entries:

        # define a custom list cotain entries files paths
        custom_list = os.listdir(os.path.join(split_path, entry))

        # list must be sorted
        custom_list.sort(key=lambda f: int(re.sub(r'\D', '', f)))

        # repeat the process for each file path
        for file_path in custom_list:

            text_output = pdf_to_text(
                os.path.join(split_path, entry, file_path))

            # save text file of each entry
            with open(os.path.join(text_path, f"{entry}.txt"),
                      "a",
                      encoding="utf-8") as text_file:
                text_file.write(text_output)