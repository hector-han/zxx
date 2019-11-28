import pandas as pd

def export_to_json_file(data_frame: pd.DataFrame, file_name: str, lines=True):
    """
    把data_frame数据导出到文件，json格式。
    :param data_frame: 数据
    :param file_name: 导出的文件名
    :param lines: 是否启用多行，默认启用。True：每一行是一个json;False:整体是个json list
    :return: None
    """
    data_frame.to_json(file_name, orient='records', lines=lines)


def read_from_json_file(file_name: str, lines=True) -> pd.DataFrame:
    """
    从json文件中读取数据到data_frame中
    :param file_name:
    :param lines: 是否启用多行
    :return: pd.DataFrame
    """
    return pd.read_json(file_name, orient='records', encoding='utf-8', lines=lines)


def export_to_excel_file(data_frame: pd.DataFrame, file_name: str):
    excel_writer = pd.ExcelWriter(file_name, engine="xlsxwriter")
    data_frame.to_excel(excel_writer)
    excel_writer.save()



def get_cate(data):
    cate = ''
    maxscore = 0.0
    for key, val in data.items():
        if val > maxscore:
            maxscore = val
            cate = key
    return cate

def create_label_xlsx():
    map1 = {
        'POSITIVE': '正',
        'CENTRAL': '中',
        'NEGATIVE': '负',
    }
    data_dir = r'D:\code\github\zxx\data'
    final_file = data_dir + r'\final.jl'
    df_final = read_from_json_file(final_file)
    df_final = df_final[['ID', 'text', 'senti', 'is_label', 'choose', 'lda5']]
    df_final = df_final[df_final['choose'] == 1]
    df_final.ID = df_final.ID.apply(lambda x: str(x))
    df_final.senti = df_final.senti.apply(lambda x: map1[x])
    df_final['cate'] = df_final.lda5.apply(get_cate)
    df_final.rename({'text': '原始推文', 'senti': '情感标注'})

    file_name = r'D:\code\github\zxx\data\情感标注.xlsx'
    excel_writer = pd.ExcelWriter(file_name, engine="xlsxwriter", options={'strings_to_urls': False})
    for cate in list('01234'):
        df_cate = df_final[df_final.cate == cate]
        df_cate = df_cate.drop(['cate', 'lda5', 'choose'], axis=1)
        df_cate.to_excel(excel_writer, sheet_name=cate)
    excel_writer.save()


if __name__ == '__main__':
    create_label_xlsx()