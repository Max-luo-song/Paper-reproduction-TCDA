import matplotlib.pyplot as plt
import matplotlib
import numpy as np
#绘图

SW_ES = [5550.1993886242, 7093.622626597412, 12711.203000549054, 17119.96954613979, 20569.029922845202, 24424.44657408173, 23269.624835395025, 27837.15995451564, 31090.609096929402, 35894.19985742175]
NMSD_ES = [21, 40, 73, 81, 100, 116, 132, 137, 158, 169]
RU_ES = [0.9101675332177932, 0.9593140031930782, 0.8392260446170002, 0.7874247203900201, 0.8055798442361266, 0.738040192068291, 0.7512510608410641, 0.6832869080779944, 0.7175156874246063, 0.6395802636294851]
def draw_ES(Y,yname):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    x = np.arange(len(Y))
    #索引列表
    numbers = []
    for i in range(20,201,20):
        numbers.append(i)
    plt.bar(x,Y,width=0.5,color='limegreen')
    #设置标题以及字体大小
    plt.title(label='TCDA',fontsize=18)
    plt.xlabel('Numbers of ESs',fontsize=14)
    plt.ylabel(yname,fontsize=14)
    plt.xticks(x,numbers)
    #设置坐标轴刻度
    plt.tick_params(axis='both',labelsize=12,color='blue' )
    plt.show()
SW_MD = [12715.645447351217, 15090.078992410097, 17005.749969073404, 19049.17165497683, 22922.718255952583, 18939.236217591537, 20939.04235778402, 20174.332470068704, 22331.531485904914, 25380.878476012753]
NMSD_MD = [56, 75, 90, 101, 115, 107, 112, 104, 105, 111]
RU_MD = [0.4660750187851587, 0.5804084675599851, 0.7173566384327045, 0.7815516691971733, 0.8406819639568807, 0.8721737405791352, 0.8611253564175675, 0.8299815652756829, 0.859634064645788, 0.925582237967753]
def draw_MD(Y,yname):
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    x = np.arange(len(Y))
    #索引列表
    numbers = []
    for i in range(100,1001,100):
        numbers.append(i)
    plt.bar(x,Y,width=0.5,color='limegreen')
    #设置标题以及字体大小
    plt.title(label='TCDA',fontsize=18)
    plt.xlabel('Numbers of MDs',fontsize=14)
    plt.ylabel(yname,fontsize=14)
    plt.xticks(x,numbers)
    #设置坐标轴刻度
    plt.tick_params(axis='both',labelsize=12,color='blue' )
    plt.show()
def draw_Participants():
    # 设置字体为微软雅黑，解决中文显示问题
    matplotlib.rc("font", family='Microsoft YaHei')
    # 1.准备数据
    S1_SW = [5978.811087079457, 7903.81282720451, 12690.924378231975, 14777.137361028468, 19056.583471785052,
             21901.70114175608, 24517.078723785868]
    S2_SW = [3851.4909887141935, 9161.691779981307, 10944.031753141424, 15178.066749318492, 21615.31258742871,
             25063.426042715244, 26760.610432292466]
    S3_SW = [4666.740346892653, 7138.790227136082, 12313.051282781638, 13272.958807550154, 18738.718849181692,
             24085.18777373821, 27489.252236820445]
    S4_SW = [3649.9200778253885, 7506.545465388259, 10836.84658739367, 16135.62573327328, 19632.474146692377,
             21543.00152408768, 30095.259181931055]
    # 索引列表
    participants = [120, 240, 360, 480, 600, 720, 840]
    """
    设置刻度：
    - xticks()：设置x轴刻度
    - yticks()：设置y轴刻度
    """
    # 设置x轴刻度
    plt.xticks(participants)
    # 设置线条样式
    plt.plot(participants, S1_SW, '-o', participants, S2_SW, '-^', participants, S3_SW, '-d', participants, S4_SW, '-s')
    # 设置标题及字体大小
    plt.title('TCDA', fontsize=20)
    plt.xlabel('Numbers of participants', fontsize=14)
    plt.ylabel('SW', fontsize=14)
    # 设置坐标轴刻度
    plt.tick_params(axis='both', labelsize=12, color='red')
    plt.legend(['S1', 'S2', 'S3', 'S4'], loc='best')
    plt.show()

def draw_Payments():
        """
        axis()设置绘图区间：
        axis([xmin, xmax, ymin, ymax])
        xmin/xmax：x轴的最小/最大区间
        ymin/ymxa：y轴的最小/最大区间
        """
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False

        x1 = [62.90589004798984, 358.04397591778115, 219.42671443146892, 270.3191432582787, 288.6536346031615,
              139.49538098474048, 346.9007519113769, 114.3550379449963, 339.60383202897117, 86.35044862223117,
              309.0012156871184, 11.909385074327812, 12.88882810890395, 63.97929532281152, 135.53753969669228,
              72.51546198682384, 474.43645251552573, 90.9799605501328, 348.6622549322712, 142.83206613433464,
              283.4787732245644, 260.09756652034304, 235.63098161396815, 177.0292891352234, 260.8247823909224,
              118.3792385559402, 8.526466948327506, 188.394766917191, 198.80874026277357, 211.16228054772066,
              44.01079748225547, 250.20917286897395, 92.01586549712049, 362.6849143227965, 65.09113771326021,
              34.50096268723483, 253.59615681294397, 180.47945962291564, 167.31825657098247, 45.96366120648882,
              311.40251315212214, 134.62135290335215, 465.5263367187486, 467.5025499879403, 182.68751039131578,
              11.914434323745851, 97.13780020029257, 382.034173932262, 77.28484017358579, 364.25386274946464,
              43.797631553885964, 139.90232577738007, 5.166701458118538, 193.303740196823, 123.89771027854955,
              359.5185077830322, 333.51884186354516, 82.40895971960126, 20.9924715051438, 71.85325844476618,
              367.7395477116692, 58.2871004637965, 37.63771772657295, 166.3318703037563, 175.25689581938977,
              49.9961036452745, 188.27101428195613, 113.04664319252497, 74.76062592930442, 267.1003256771035,
              463.7800389933899, 268.08540392080704, 283.7043406932714, 282.8771688171234, 322.5163242882745,
              144.17003417550404, 43.54776140437592, 269.57880828776337, 366.7384827892112, 78.2742437227134,
              55.62959072225037, 335.34579174003477, 246.04077018582345, 263.58333701665714, 194.11143816057836,
              115.5761846786668, 311.0465158119341, 91.28579834543224, 152.37552451602477, 410.5032390715096,
              134.5982562783456, 163.25225798161054, 357.73925521811583, 276.51991899664034, 221.29859513597424,
              216.93149248903444, 317.11553746874176, 255.73425582849097, 4.269771353942296, 6.882256617204256]
        y1 = [62.90589004798984, 358.04397591778115, 219.42671443146892, 270.3191432582787, 288.6536346031615,
              139.49538098474048, 346.9007519113769, 114.3550379449963, 339.60383202897117, 86.35044862223117,
              309.0012156871184, 11.909385074327812, 12.88882810890395, 63.97929532281152, 135.53753969669228,
              72.51546198682384, 474.43645251552573, 90.9799605501328, 348.6622549322712, 142.83206613433464,
              283.4787732245644, 260.09756652034304, 235.63098161396815, 177.0292891352234, 260.8247823909224,
              118.3792385559402, 8.526466948327506, 188.394766917191, 198.80874026277357, 211.16228054772066,
              44.01079748225547, 250.20917286897395, 92.01586549712049, 362.6849143227965, 65.09113771326021,
              34.50096268723483, 253.59615681294397, 180.47945962291564, 167.31825657098247, 45.96366120648882,
              311.40251315212214, 134.62135290335215, 465.5263367187486, 467.5025499879403, 182.68751039131578,
              11.914434323745851, 97.13780020029257, 382.034173932262, 77.28484017358579, 364.25386274946464,
              43.797631553885964, 139.90232577738007, 5.166701458118538, 193.303740196823, 123.89771027854955,
              359.5185077830322, 333.51884186354516, 82.40895971960126, 20.9924715051438, 71.85325844476618,
              367.7395477116692, 58.2871004637965, 37.63771772657295, 166.3318703037563, 175.25689581938977,
              49.9961036452745, 188.27101428195613, 113.04664319252497, 74.76062592930442, 267.1003256771035,
              463.7800389933899, 268.08540392080704, 283.7043406932714, 282.8771688171234, 322.5163242882745,
              144.17003417550404, 43.54776140437592, 269.57880828776337, 366.7384827892112, 78.2742437227134,
              55.62959072225037, 335.34579174003477, 246.04077018582345, 263.58333701665714, 194.11143816057836,
              115.5761846786668, 311.0465158119341, 91.28579834543224, 152.37552451602477, 410.5032390715096,
              134.5982562783456, 163.25225798161054, 357.73925521811583, 276.51991899664034, 221.29859513597424,
              216.93149248903444, 317.11553746874176, 255.73425582849097, 4.269771353942296, 6.882256617204256]
        x2 = [62.90589004798984, 358.04397591778115, 219.42671443146892, 270.3191432582787, 288.6536346031615,
              139.49538098474048, 346.9007519113769, 114.3550379449963, 339.60383202897117, 86.35044862223117,
              309.0012156871184, 11.909385074327812, 12.88882810890395, 63.97929532281152, 135.53753969669228,
              72.51546198682384, 474.43645251552573, 90.9799605501328, 348.6622549322712, 142.83206613433464,
              283.4787732245644, 260.09756652034304, 235.63098161396815, 177.0292891352234, 260.8247823909224,
              118.3792385559402, 8.526466948327506, 188.394766917191, 198.80874026277357, 211.16228054772066,
              44.01079748225547, 250.20917286897395, 92.01586549712049, 362.6849143227965, 65.09113771326021,
              34.50096268723483, 253.59615681294397, 180.47945962291564, 167.31825657098247, 45.96366120648882,
              311.40251315212214, 134.62135290335215, 465.5263367187486, 467.5025499879403, 182.68751039131578,
              11.914434323745851, 97.13780020029257, 382.034173932262, 77.28484017358579, 364.25386274946464,
              43.797631553885964, 139.90232577738007, 5.166701458118538, 193.303740196823, 123.89771027854955,
              359.5185077830322, 333.51884186354516, 82.40895971960126, 20.9924715051438, 71.85325844476618,
              367.7395477116692, 58.2871004637965, 37.63771772657295, 166.3318703037563, 175.25689581938977,
              49.9961036452745, 188.27101428195613, 113.04664319252497, 74.76062592930442, 267.1003256771035,
              463.7800389933899, 268.08540392080704, 283.7043406932714, 282.8771688171234, 322.5163242882745,
              144.17003417550404, 43.54776140437592, 269.57880828776337, 366.7384827892112, 78.2742437227134,
              55.62959072225037, 335.34579174003477, 246.04077018582345, 263.58333701665714, 194.11143816057836,
              115.5761846786668, 311.0465158119341, 91.28579834543224, 152.37552451602477, 410.5032390715096,
              134.5982562783456, 163.25225798161054, 357.73925521811583, 276.51991899664034, 221.29859513597424,
              216.93149248903444, 317.11553746874176, 255.73425582849097, 4.269771353942296, 6.882256617204256]
        y2 = [0, 309.39334644480766, 0, 0, 274.4316655908854, 0, 312.34618482645465, 0, 280.244557195697, 0,
              274.6006897219509, 0, 0, 0, 0, 0, 390.1127861504616, 0, 316.2941875657061, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              0, 0, 0, 0, 336.05852436461475, 0, 0, 0, 0, 0, 0, 0, 0, 401.9709267440784, 398.1945493061603, 0, 0, 0, 0,
              0, 321.8564298828292, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 300.9431064281042, 0, 0, 0, 0, 0, 0, 0, 0, 0,
              387.7853059573779, 0, 242.03786758119736, 0, 285.0915278398668, 0, 0, 0, 0, 0, 0, 326.83115249663547, 0,
              0, 0, 0, 0, 0, 0, 330.52506105154384, 0, 0, 355.24570780387637, 0, 0, 0, 0, 242.74775064970044, 0, 0]
        # 设置绘图区间
        plt.axis([200, 500, 200, 500])
        plt.scatter(x=x1, y=y1, s=8, c='red')
        plt.scatter(x=x2, y=y2, s=8, c='blue')
        # 设置标题及字体大小
        plt.title('TCDA', fontsize=20)
        plt.xlabel('Bidding prices', fontsize=14)
        plt.ylabel('Payments & bidding prices', fontsize=14)
        plt.legend(['Bidding prices', 'Payments'], loc='best')
        # 设置坐标轴刻度
        plt.tick_params(axis='both', labelsize=12, color='red')
        plt.show()

def draw_Rewards():
        plt.rcParams['font.sans-serif'] = ['SimHei']
        plt.rcParams['axes.unicode_minus'] = False
        x1 = [154.18940357785013, 0.0, 61.37106253212624, 0.0, 0.0, 0.0, 63.535929835543996, 85.79077230903577, 0.0,
         188.90149355437208, 0.0, 140.23403598146317, 230.70405930729987, 104.70036168430943, 0.0, 173.49359627099457,
         88.96402864901651, 214.01996645038247, 31.334987509832096, 71.25298925843623, 34.77130177193802,
         73.75953628829026, 244.25733725233226, 190.48592073501698, 0.0, 32.411104354390844, 156.65165526401626,
         104.34156502613844, 45.82160621544982, 99.48449658820816, 133.37834288727566, 0.0, 0.0, 0.0, 0.0,
         4.1938777656511075, 52.91969055694955, 126.34445406614226, 191.06901280026395, 134.32244546993445,
         254.8848292355946, 164.75305931280872, 292.39902398077515, 411.84198664241154, 178.95715732166107, 0.0,
         177.53895361263613, 251.88324956498522, 0.0, 50.763740068596036]
        y1 = [154.18940357785013, 0.0, 61.37106253212624, 0.0, 0.0, 0.0, 63.535929835543996, 85.79077230903577, 0.0,
              188.90149355437208, 0.0, 140.23403598146317, 230.70405930729987, 104.70036168430943, 0.0,
              173.49359627099457,
              88.96402864901651, 214.01996645038247, 31.334987509832096, 71.25298925843623, 34.77130177193802,
              73.75953628829026, 244.25733725233226, 190.48592073501698, 0.0, 32.411104354390844, 156.65165526401626,
              104.34156502613844, 45.82160621544982, 99.48449658820816, 133.37834288727566, 0.0, 0.0, 0.0, 0.0,
              4.1938777656511075, 52.91969055694955, 126.34445406614226, 191.06901280026395, 134.32244546993445,
              254.8848292355946, 164.75305931280872, 292.39902398077515, 411.84198664241154, 178.95715732166107, 0.0,
              177.53895361263613, 251.88324956498522, 0.0, 50.763740068596036]
        x2 = [154.18940357785013, 0.0, 61.37106253212624, 0.0, 0.0, 0.0, 63.535929835543996, 85.79077230903577, 0.0,
              188.90149355437208, 0.0, 140.23403598146317, 230.70405930729987, 104.70036168430943, 0.0,
              173.49359627099457,
              88.96402864901651, 214.01996645038247, 31.334987509832096, 71.25298925843623, 34.77130177193802,
              73.75953628829026, 244.25733725233226, 190.48592073501698, 0.0, 32.411104354390844, 156.65165526401626,
              104.34156502613844, 45.82160621544982, 99.48449658820816, 133.37834288727566, 0.0, 0.0, 0.0, 0.0,
              4.1938777656511075, 52.91969055694955, 126.34445406614226, 191.06901280026395, 134.32244546993445,
              254.8848292355946, 164.75305931280872, 292.39902398077515, 411.84198664241154, 178.95715732166107, 0.0,
              177.53895361263613, 251.88324956498522, 0.0, 50.763740068596036]
        y2 = [291.03638965500977, 0, 190.69411700695673, 0, 0, 0, 203.3880189684587, 124.2022495895726, 0, 217.8953354958885,
         0, 142.65527918638873, 289.22297508908196, 242.6786678969147, 0, 235.425009633198, 199.76118983660126,
         430.06483970954105, 171.95549982569537, 393.7965483909684, 173.16444286964725, 208.82826266624397,
         391.3786623030628, 301.3124055286062, 0, 129.64249328735968, 288.0140320451301, 249.93232616062778,
         381.1026464294682, 156.84371177628964, 414.3485801381594, 0, 0, 0, 0, 321.8644372757972, 371.43110207784775,
         259.60387051224643, 348.4611842427512, 152.0079396004785, 301.9168770505821, 256.5815129023649,
         300.1034624846543, 445.7810992809227, 446.38557080289866, 0, 289.22297508908196, 451.2213429787098, 0,
         254.16362681446117]
        plt.axis([0, 600, 0, 600])
        plt.scatter(x=x1, y=y1, s=8, c='red')
        plt.scatter(x=x2, y=y2, s=8, c='blue')
        # 设置标题及字体大小
        plt.title('TCDA', fontsize=20)
        plt.xlabel('Asking prices', fontsize=14)
        plt.ylabel('Rewards & asking prices', fontsize=14)
        plt.legend(['Asking prices', 'Rewards'], loc='best')
        # 设置坐标轴刻度
        plt.tick_params(axis='both', labelsize=12, color='red')
        plt.show()
def draw_Budget_Balance():
    Payments = [3205.3898649002035, 4383.084893633365, 7095.174679958718, 8013.366017444753, 9506.068662623353,
                8309.843545035661, 9616.775168484928, 9812.266738865812, 10352.830855976605, 9984.06637017842]
    rewards = [2212.4460207534985, 3895.3221661465204, 5957.109902414642, 7812.351688775224, 8746.157616184619,
               8270.09565073322, 9532.194381793222, 9615.805201751062, 10333.88872246989, 9221.419775742455]
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    x = np.arange(len(Payments))
    # 索引列表
    numbers = [8, 16, 24, 32, 40, 48, 56, 64, 72, 80]
    plt.bar(x, Payments, width=0.5, color='blue')
    plt.bar(x, rewards, width=0.3, color='red')
    # 设置标题以及字体大小
    plt.title(label='TCDA', fontsize=18)
    plt.xlabel('Numbers of ESs', fontsize=14)
    plt.ylabel('Payments & rewards', fontsize=14)
    plt.xticks(x, numbers)
    # 设置坐标轴刻度
    plt.legend(['Payments from MDs', 'Rewards to ESs'], loc='best')
    plt.tick_params(axis='both', labelsize=12, color='red')
    plt.show()

MD_Utility_win = [0, 0, 0, 0, -0.375, 1.0625, 2.25, 3.0625, 4.375, 4.6875, 6.0, 6.90625, 7.4375]
MD_Bidding_price_win = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18]
MD_Utility_lose = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
MD_Bidding_price_lose = [7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21]
ES_Utility_win = [7.099999999999998, 6.399999999999997, 5.699999999999992, 4.9999999999999964,
                  4.299999999999994, 3.5999999999999943, 2.899999999999995, 2.199999999999992,
                  1.499999999999993, 1.0, 0.5, -3.1086244689504383e-15, 0.0, 0.0, 0.0, 0.0,
                  0.0, 0.0, 0.0, 0.0, 0.0]
ES_Asking_price_win = [2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,4.0]
ES_Utility_lose = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
ES_Asking_price_lose = [2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3.0, 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8, 3.9,4.0]
def draw_MD_Utility(Utility,Bidding_price):
    # 设置字体为微软雅黑，解决中文显示问题
    matplotlib.rc("font", family='Microsoft YaHei')
    # 1.准备数据
    # 索引列表
    """
    设置刻度：
    - xticks()：设置x轴刻度
    - yticks()：设置y轴刻度
    """
    # 设置x轴刻度
    plt.xticks(Bidding_price)
    # 设置线条样式
    plt.plot(Bidding_price, Utility, '-o')
    # 设置标题及字体大小
    plt.title('TCDA', fontsize=20)
    plt.xlabel('Bidding_price', fontsize=14)
    plt.ylabel('Utility', fontsize=14)
    # 设置坐标轴刻度
    plt.tick_params(axis='both', labelsize=12, color='red')
    plt.show()

def draw_ES_Utility(Utility,Asking_price):
    # 设置字体为微软雅黑，解决中文显示问题
    matplotlib.rc("font", family='Microsoft YaHei')
    # 1.准备数据
    # 索引列表
    """
    设置刻度：
    - xticks()：设置x轴刻度
    - yticks()：设置y轴刻度
    """
    # 设置x轴刻度
    plt.xticks(Asking_price)
    # 设置线条样式
    plt.plot(Asking_price, Utility, '-o')
    # 设置标题及字体大小
    plt.title('TCDA', fontsize=20)
    plt.xlabel('Asking_price', fontsize=14)
    plt.ylabel('Utility', fontsize=14)
    # 设置坐标轴刻度
    plt.tick_params(axis='both', labelsize=6, color='red')
    plt.show()
if __name__ == '__main__':
    draw_ES(SW_ES,'SW')
    draw_ES(NMSD_ES,'NMSD')
    draw_ES(RU_ES,'RU')
    draw_MD(SW_MD,'SW')
    draw_MD(NMSD_MD,'NMSD')
    draw_MD(RU_MD,'RU')
    draw_Participants()
    draw_Payments()
    draw_Rewards()
    draw_Budget_Balance()
    draw_MD_Utility(MD_Utility_win,MD_Bidding_price_win)
    draw_MD_Utility(MD_Utility_lose,MD_Bidding_price_lose)
    draw_ES_Utility(ES_Utility_win,ES_Asking_price_win)
    draw_ES_Utility(ES_Utility_lose,ES_Asking_price_lose)