def missing_ratio(data):

    # how many columns the data have:
    import pandas as pd

    print("----------------------")

    col = data.shape[1]
    row = data.shape[0]
    flag = 0

    for i in range(col):
        count = 0
        ratio = 0
        for j in range(row):
            if pd.isnull(data.iloc[j,i]) == True:
                count = count + 1
            ratio = count/row
        if ratio != 0:
            print("column " + str(i) + " has " + str(ratio) + "% missing values.")
            flag = 1

    if flag == 0:
        print("Congratulations! No missing values! ")
