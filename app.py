import streamlit as st
import pandas as pd
import docx
import numpy as np

from checkwp import wpreview
from utils import display_entities,get_ner_labels
from corrector import highlight_word,tup2list


def main():

    # choose input method of manual or upload file
    input_method = st.sidebar.radio('Input Method', ('Manual', 'Upload File'))

    if input_method == 'Manual':
        proc_text = st.text_area('Testing Procedures')
        audit_text = st.text_area('Testing Descriptions')

        proc_list = proc_text.split('/')
        # filter blank item
        proc_list = list(filter(lambda item: item.strip(), proc_list))
        audit_list = audit_text.split('/')
        # filter blank item
        audit_list = list(filter(lambda item: item.strip(), audit_list))

    elif input_method == 'Upload File':
        # upload file
        upload_file = st.file_uploader('Upload workpaper', type=['xlsx','docx'])
        if upload_file is not None:
            # if upload file is xlsx
            if upload_file.type == 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet':
                # get sheet names list from excel file
                xls = pd.ExcelFile(upload_file)
                sheets = xls.sheet_names
                # choose sheet name and click button
                sheet_name = st.selectbox('Choose sheetname', sheets)

                # choose header row
                header_row = st.number_input('Choose header row',
                                            min_value=0,
                                            max_value=10,
                                            value=0)
                df = pd.read_excel(upload_file,
                                header=header_row,
                                sheet_name=sheet_name)
                # filllna
                df = df.fillna('')
                # display the first five rows
                st.write(df.astype(str))

                # get df columns
                cols = df.columns
                # choose proc_text and audit_text column
                proc_col = st.sidebar.selectbox('Choose procedure column', cols)
                audit_col = st.sidebar.selectbox('Choose testing column', cols)
                # get proc_text and audit_text list
                proc_list = df[proc_col].tolist()
                audit_list = df[audit_col].tolist()

            elif upload_file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                # read docx file
                document = docx.Document(upload_file)
                # get table data
                tablels=[]
                for table in document.tables:
                    tb=[]
                    for row in table.rows:
                        rl=[]
                        for cell in row.cells:
                            rl.append(cell.text)
                        tb.append(rl)
                    tablels.append(tb)

                # get tablels index list
                tablels_index = list(range(len(tablels)))
                # choose tablels index
                tablels_no = st.selectbox('Choose table index', tablels_index)
                # choose header row
                header_row = st.number_input('Choose header row',
                                            min_value=0,
                                            max_value=10,
                                            value=0)

                if tablels_no is not None:
                    # get tablels
                    data = tablels[tablels_no]
                    dataarray=np.array(data)
                    dataarray2=dataarray[header_row:,:]
                    df=pd.DataFrame(dataarray2)

                    st.write(df.astype(str))
    
                    # get df columns
                    cols = df.columns
                    # choose proc_text and audit_text column
                    proc_col = st.sidebar.selectbox('Choose procedure column', cols)
                    audit_col = st.sidebar.selectbox('Choose testing column', cols)
                    # get proc_text and audit_text list
                    proc_list = df[proc_col].tolist()
                    audit_list = df[audit_col].tolist()
                else:
                    st.error('No table found in the document')
                    proc_list = []
                    audit_list = []
        else:
            st.error('No file selected')
            proc_list = []
            audit_list = []

    # get proc_list and audit_list length
    proc_len = len(proc_list)
    audit_len = len(audit_list)

    # if proc_list or audit_list is empty or not equal
    if proc_len == 0 or audit_len == 0 or proc_len != audit_len:
        st.error('Procedure and Testing list must be equal and not empty')
        st.error(
            '(Testing Procedure:' +
            str(len(proc_list)) + '/ Testing Description:' +
            str(len(audit_list)) + ')')
        return

    with st.sidebar.expander('Parameters'):
        # choose parameters
        threshold = st.slider('Sentence Matching Threshold',
                            min_value=0.0,
                            max_value=1.0,
                            value=0.8,
                            key='sentence')
        
        st.write('Sentence Threshold:', threshold)
        threshold_key = st.slider('Keyword Matching Threshold',
                            min_value=0.0,
                            max_value=1.0,
                            value=0.6,
                            key='key')

        st.write('Keyword Threshold:', threshold_key)
        top = st.slider('Keyword Number',
                                min_value=1,
                                max_value=10,
                                value=5)
        # get ner labels
        ner_labels = get_ner_labels()
        # choose ner label using multi-select
        ner_label = st.multiselect('Choose NER label', ner_labels,ner_labels)

        # choose start and end index
        start_idx = st.number_input('Choose start index',
                                min_value=0,
                                max_value=proc_len-1,
                                value=0)
        # convert start_idx to int
        start_idx = int(start_idx)
        end_idx = st.number_input('Choose end index',
                            min_value=start_idx,
                            max_value=proc_len-1,
                            value=proc_len-1)
        # convert end_idx to int
        end_idx = int(end_idx)
        # get proc_list and audit_list
        subproc_list = proc_list[start_idx:end_idx+1]
        subaudit_list = audit_list[start_idx:end_idx+1]

    search = st.sidebar.button('Review')
  
    if search:
        # split list into batch of 5
        batch_num = 5
        proc_list_batch = [
            subproc_list[i:i + batch_num]
            for i in range(0, len(subproc_list), batch_num)
        ]
        audit_list_batch = [
            subaudit_list[i:i + batch_num]
            for i in range(0, len(subaudit_list), batch_num)
        ]

        dfls = []
        # get proc and audit batch
        for j, (proc_batch, audit_batch) in enumerate(
                zip(proc_list_batch, audit_list_batch)):

            with st.spinner('Reviewing...'):
                # range of the batch
                start = j * batch_num + 1
                end = start + len(proc_batch) - 1
                st.subheader('Overview: ' + f'{start}-{end}')
                dfsty, df, highlight_proc, highlight_audit, distancels, emptyls, proc_keywords, errorls = wpreview(
                    proc_batch, audit_batch, threshold, threshold_key, top)

                # display the result
                st.write(dfsty)

                st.subheader('Content: ' + f'{start}-{end}')
                for i, (proc, audit, distance, empty, keywordls, error,
                        proc_text, audit_text) in enumerate(
                            zip(highlight_proc, highlight_audit,
                                distancels, emptyls, proc_keywords,
                                errorls, proc_batch, audit_batch)):
                    count = str(j * batch_num + i + 1)
                    st.warning('Procedure ' + count + ': ')
                    display_entities(proc_text, str(count) + '_proc',ner_label)

                    # keywords list to string
                    keywords = ','.join(keywordls)
                    st.markdown('Keyword:' + keywords)

                    st.warning('Testing Description ' + count + ': ')
                    display_entities(audit_text, str(count) + '_audit',ner_label)

                    st.warning('Review Result ' + count + ': ')
                    # combine empty list to text
                    if empty:
                        empty_text = ' '.join(empty)
                        st.error('Missing Keyword: ' + empty_text)
                    else:
                        st.success('No Missing Keyword')

                    # combine error list to text

                    if error:
                        error_text=tup2list(error)
                        st.error('Error Found: '+error_text)
                        for detail in error:
                            corrected_hl = list(audit_text)
                            corrected_hl = highlight_word(corrected_hl, detail[2],
                                                            detail[3],detail[1])

                            corrected_hlstr = ''.join(corrected_hl)
                            st.markdown('Corrected text: ' + corrected_hlstr,
                                        unsafe_allow_html=True)
                    else:
                        st.success('No error found in text')

                    if distance >= threshold:
                        st.success('Pass: ' + str(distance))
                    else:
                        st.error('Fail: ' + str(distance))
                dfls.append(df)

        # review is done
        st.sidebar.success('Review Finish')
        # download df
        alldf = pd.concat(dfls)
        st.sidebar.download_button(label='Download',
                                    data=alldf.to_csv(),
                                    file_name='WPreviewresult.csv',
                                    mime='text/csv')


if __name__ == '__main__':
    main()