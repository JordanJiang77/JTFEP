from xml.dom.minidom import parse
import xml.dom.minidom
import pandas as pd

import os


def intinvert4str(a):
    if a <= 9:
        return "000" + str(a)
    elif a <= 99:
        return "00" + str(a)
    elif a <= 999:
        return "0" + str(a)
    else:
        return str(a)


def findNumdirect01(vdid, dataOneTime):
    for indexx, oneData in dataOneTime.iterrows():
        if (oneData[3] == vdid):
            return oneData[0], oneData[1]


# 存静态的道路名称和vdid于dataPathXY中
dataPathXY = pd.DataFrame()

for time in range(0, 1):
    # if os.path.exists("data_xml/vd_info_" + intinvert4str(time) + ".xml") == False: continue
    # print("data_xml/vd_info_" + intinvert4str(time) + ".xml")

    DOMTree = xml.dom.minidom.parse("data_xml/vd_info_0000.xml")

    XML_Head = DOMTree.documentElement

    InfoSet = XML_Head.getElementsByTagName("Infos")

    for infoSet in InfoSet:
        oneInfoSet = infoSet.getElementsByTagName("Info")
        for oneInfo in oneInfoSet:
            vdid = oneInfo.getAttribute("vdid")
            routeid = oneInfo.getAttribute("routeid")
            roadsection = oneInfo.getAttribute("roadsection")
            indexLeftKuo = roadsection.find('(', 1)
            roadsection = roadsection[0:indexLeftKuo]
            px = float(oneInfo.getAttribute("px"))
            py = float(oneInfo.getAttribute("py"))
            oneData = {"vdid": vdid, "routeid": routeid, "roadsection": roadsection, "px": px, "py": py}
            dataPathXY = dataPathXY.append(oneData, ignore_index=True)

# 选择需要的数据
dataChoose = pd.DataFrame()
lastPath = dataPathXY.iloc[0][2]
# 每一轮按照时间顺序存190个vdid
data = pd.DataFrame()
for time in range(0, 20, 5):
    if os.path.exists("data_xml/vd_value_" + intinvert4str(time) + ".xml") == False: continue
    print("data_xml/vd_value_" + intinvert4str(time) + ".xml")

    DOMTree = xml.dom.minidom.parse("data_xml/vd_value_" + intinvert4str(time) + ".xml")

    XML_Head = DOMTree.documentElement

    InfoSet = XML_Head.getElementsByTagName("Infos")

    for infoSet in InfoSet:
        standardLines = 0;
        oneInfoSet = infoSet.getElementsByTagName("Info")
        for oneInfo in oneInfoSet:
            vdid = oneInfo.getAttribute("vdid")
            LaneSet = oneInfo.getElementsByTagName("lane")
            numDirect0 = 0
            numDirect1 = 0
            for lane in LaneSet:
                vsrdir = lane.getAttribute('vsrdir')
                if (vsrdir == '0'):
                    for car in lane.getElementsByTagName("cars"):
                        standardLines += 1;
                        volume = car.getAttribute('volume')
                        numDirect0 += int(volume)
                if (vsrdir == '1'):
                    for car in lane.getElementsByTagName("cars"):
                        standardLines += 1;
                        volume = car.getAttribute('volume')
                        numDirect1 += int(volume)
    print(standardLines)
    if (standardLines <= 1400): continue

    InfoSet = XML_Head.getElementsByTagName("Infos")

    for infoSet in InfoSet:
        # 存一个时间的vdid，用来给平均值遍历使用
        dataOnetime = pd.DataFrame()
        standardLines = 0;
        oneInfoSet = infoSet.getElementsByTagName("Info")
        for oneInfo in oneInfoSet:
            vdid = oneInfo.getAttribute("vdid")
            LaneSet = oneInfo.getElementsByTagName("lane")
            numDirect0 = 0
            numDirect1 = 0
            for lane in LaneSet:
                vsrdir = lane.getAttribute('vsrdir')
                if (vsrdir == '0'):
                    for car in lane.getElementsByTagName("cars"):
                        standardLines += 1;
                        volume = car.getAttribute('volume')
                        numDirect0 += int(volume)
                if (vsrdir == '1'):
                    for car in lane.getElementsByTagName("cars"):
                        standardLines += 1;
                        volume = car.getAttribute('volume')
                        numDirect1 += int(volume)
            oneData = {"vdid": vdid, "numDirect0": numDirect0, "numDirect1": numDirect1, "time": time}
            dataOnetime = dataOnetime.append(oneData, ignore_index=True)
            data = data.append(oneData, ignore_index=True)
        # 一个时间的vdid已经存入data中
        # 开始做数据的列表
        numDirect0, numDirect1 = findNumdirect01(dataPathXY.iloc[0][4], dataOnetime)
        lastPath = dataPathXY.iloc[0][2]
        oneData = {"time": time, "px": dataPathXY.iloc[0][0], "py": dataPathXY.iloc[0][1],
                   "roadsection": lastPath, "numDirect0": numDirect0, "numDirect1": numDirect1}
        for index, row in dataPathXY.iterrows():
            lastPath = row[2]
            if (lastPath == "凱旋路" or lastPath == "中正路" or lastPath == "五福路" or lastPath == "民族路" or
            lastPath == "民權路" or lastPath == "三多路"):
                numDirect0, numDirect1 = findNumdirect01(row[4], dataOnetime)
                oneData = {"time": time, "roadsection": lastPath,"numDirect0": numDirect0, "numDirect1": numDirect1}
        # 最后一组
                dataChoose = dataChoose.append(oneData, ignore_index=True)
                pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
print(dataChoose)



