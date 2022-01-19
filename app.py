import streamlit as st
import pandas as pd
import docx
import numpy as np

from checkwp import wpreview
from utils import display_entities
from corrector import highlight_word,tup2list


def main():

    # choose input method of manual or upload file
    input_method = st.sidebar.radio('Input Method', ('Manual', 'Upload File'))

    if input_method == 'Manual':
        proc_text = st.text_area('Testing Procedures')
        audit_text = st.text_area('Testing Descriptions')

        proc_list = proc_text.split()
        # filter blank item
        proc_list = list(filter(lambda item: item.strip(), proc_list))
        audit_list = audit_text.split()
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
                # choose header row
                header_row = st.number_input('Choose header row',
                                            min_value=0,
                                            max_value=10,
                                            value=0)
                # get table data
                data=[]
                for table in document.tables:
                    tb=[]
                    for row in table.rows:
                        rl=[]
                        for cell in row.cells:
                            rl.append(cell.text)
                        tb.append(rl)
                    data.append(tb)
                dataarray=np.array(data)
                shape=dataarray.shape
                dataarray1=dataarray.reshape(shape[0]*shape[1],shape[2])

                dataarray2=dataarray1[header_row:,:]
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

    threshold = st.sidebar.slider('Sentence Matching Threshold',
                          min_value=0.0,
                          max_value=1.0,
                          value=0.8,
                          key='sentence')
    
    st.sidebar.write('Sentence Threshold:', threshold)
    threshold_key = st.sidebar.slider('Keyword Matching Threshold',
                          min_value=0.0,
                          max_value=1.0,
                          value=0.6,
                          key='key')

    st.sidebar.write('Keyword Threshold:', threshold_key)
    top = st.sidebar.slider('Keyword Number',
                            min_value=1,
                            max_value=10,
                            value=5)

    search = st.sidebar.button('Review')

    if search:
        # compare lengths of the two lists and list not empty
        if len(proc_list) > 0 and len(audit_list) > 0 and len(
                proc_list) == len(audit_list):
            # split list into batch of 5
            batch_num = 5
            proc_list_batch = [
                proc_list[i:i + batch_num]
                for i in range(0, len(proc_list), batch_num)
            ]
            audit_list_batch = [
                audit_list[i:i + batch_num]
                for i in range(0, len(audit_list), batch_num)
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
                        st.warning('Procedure' + count + ': ')
                        display_entities(proc_text, str(count) + '_proc')

                        # keywords list to string
                        keywords = ','.join(keywordls)
                        st.markdown('Keyword:' + keywords)

                        st.warning('Testing Description' + count + ': ')
                        display_entities(audit_text, str(count) + '_audit')

                        st.warning('Review Result' + count + ': ')
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
        else:
            st.warning('Please check your input')
            st.error(
                'Input is blank or inconsistent length(Testing Procedure:' +
                str(len(proc_list)) + ' Testing Description:' +
                str(len(audit_list)) + ')')


if __name__ == '__main__':
    main()