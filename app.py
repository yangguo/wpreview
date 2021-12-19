import streamlit as st
import pandas as pd

from checkwp import wpreview
from utils import display_entities


def main():

    # choose input method of manual or upload file
    input_method = st.sidebar.radio('选择输入方式', ('手动输入', '上传文件'))

    if input_method == '手动输入':
        proc_text = st.text_area('输入测试步骤列表')
        audit_text = st.text_area('输入现状描述列表')

        proc_list = proc_text.split()
        # filter blank item
        proc_list = list(filter(lambda item: item.strip(), proc_list))
        audit_list = audit_text.split()
        # filter blank item
        audit_list = list(filter(lambda item: item.strip(), audit_list))

    elif input_method == '上传文件':
        # upload excel file
        upload_file = st.file_uploader('上传审计底稿', type=['xlsx'])
        if upload_file is not None:
            # get sheet names list from excel file
            xls = pd.ExcelFile(upload_file)
            sheets = xls.sheet_names
            # choose sheet name and click button
            sheet_name = st.selectbox('选择工作表', sheets)

            # choose header row
            header_row = st.number_input('选择表头行',
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
            proc_col = st.sidebar.selectbox('选择测试步骤字段', cols)
            audit_col = st.sidebar.selectbox('选择现状描述字段', cols)
            # get proc_text and audit_text list
            proc_list = df[proc_col].tolist()
            audit_list = df[audit_col].tolist()

    x = st.sidebar.slider('语句匹配阈值%',
                          min_value=0,
                          max_value=100,
                          value=80,
                          key='sentence')
    threshold = x / 100
    st.sidebar.write('语句阈值:', threshold)
    y = st.sidebar.slider(' 关键词匹配阈值%',
                          min_value=0,
                          max_value=100,
                          value=60,
                          key='key')
    threshold_key = y / 100
    st.sidebar.write('关键词阈值:', threshold_key)
    top = st.sidebar.slider('关键词数量选择', min_value=1, max_value=10, value=5)

    search = st.sidebar.button('差异检查')

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

                with st.spinner('正在检查...'):
                    # range of the batch
                    start = j * batch_num + 1
                    end = start + len(proc_batch) - 1
                    st.subheader('整体检查：'+f'第{start}-{end}条')
                    dfsty, df, highlight_proc, highlight_audit, distancels, emptyls, proc_keywords = wpreview(
                        proc_batch, audit_batch, threshold, threshold_key, top)

                    # display the result
                    st.dataframe(dfsty)
                    
                    st.subheader('内容检查：' + f'第{start}-{end}条')
                    for i, (proc, audit, distance, empty, keywordls, proc_text,
                            audit_text) in enumerate(
                                zip(highlight_proc, highlight_audit,
                                    distancels, emptyls, proc_keywords,
                                    proc_batch, audit_batch)):
                        count = str(j * batch_num + i + 1)
                        st.warning('测试步骤' + count + ': ')
                        display_entities(proc_text, str(count) + '_proc')

                        # keywords list to string
                        keywords = ','.join(keywordls)
                        st.markdown('关键词：' + keywords)

                        st.warning('现状描述' + count + ': ')
                        display_entities(audit_text, str(count) + '_audit')

                        st.warning('检查结果' + count + ': ')
                        # combine empty list to text
                        empty_text = ' '.join(empty)
                        st.error('缺失内容: ' + empty_text)

                        if distance >= threshold:
                            st.success('通过: ' + str(distance))
                        else:
                            st.error('不通过: ' + str(distance))
                    dfls.append(df)

            # review is done
            st.sidebar.success('检查完成')
            # download df
            alldf = pd.concat(dfls)
            st.sidebar.download_button(label='下载结果',
                                       data=alldf.to_csv(),
                                       file_name='审计底稿检查.csv',
                                       mime='text/csv')
        else:
            st.warning('请检查输入')
            st.error('输入列表长度不一致或空值(测试步骤:' + str(len(proc_list)) + ' 现状描述:' +
                     str(len(audit_list)) + ')')


if __name__ == '__main__':
    main()