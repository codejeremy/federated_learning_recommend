import math
import numpy as np
from phe import paillier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
#设置各参与方的父类，各个参与方都需要保存模型的参数、一些中间计算结果以及与其他参与方的连接状况

class Client:
    def __init__(self,config):
        # 模型参数
        self.config = config
        # 中间 计算结果
        self.data = {}
        # 与其他节点连接状况
        self.other_client = {}
    # 与其他参与方建立连接
    def connect(self,client_name,target_client):
        print('与其他参与方建立连接')
        self.other_client[client_name] = target_client
    # 向特定参与方发送数据
    def send_data(self,data,target_client):
        print('向参与方发送数据',target_client)
        target_client.data.update(data)
#参与方A在训练的过程中仅提供特征数据
class ClientA(Client):
    def __init__(self,X,config):
        super().__init__(config)
        self.X = X
        self.weights = np.random.uniform(-0.1, 0.1, X.shape[1])
        # self.weights = np.zeros(X.shape[1])
    def compute_z_a(self):
        z_a = np.dot(self.X,self.weights)
        return z_a
    #加密梯度的计算，对应step4
    def compute_encrypted_dJ_a(self,encrypted_u):
        print('Astep4')
        encrypted_dJ_a = self.X.T.dot(encrypted_u)+self.config['lambda']*self.weights
        return encrypted_dJ_a
    #参数更新
    def update_weight(self,dJ_a):
        print('参数更新')
        self.weights=self.weights-self.config["lr"]*dJ_a/len(self.X)
        return
    # A:step2
    def task_1(self,client_B_name):
        print('Astep2')
        dt = self.data
        assert "public_key" in dt.keys(),"Error: 'public_key' from C in step 1 not successfully received."
        public_key = dt['public_key']#拿到C给A的公钥
        z_a = self.compute_z_a()
        u_a = 0.25*z_a
        z_a_square = z_a**2
        encrypted_u_a = np.asarray([public_key.encrypt(x) for x in u_a])
        encrypted_z_a_square = np.asarray([public_key.encrypt(x) for x in z_a_square])
        dt.update({"encrypted_u_a": encrypted_u_a})
        data_to_B = {"encrypted_u_a": encrypted_u_a, "encrypted_z_a_square": encrypted_z_a_square}
        print('Astep2:',len(data_to_B['encrypted_u_a']))
        self.send_data(data_to_B, self.other_client[client_B_name])
    #A:step3、4
    def task_2(self,client_C_name):
        print('Astep3、4')
        dt = self.data
        assert "encrypted_u_b" in dt.keys(), "Error: 'encrypted_u_b' from B in step 1 not successfully received."
        encrypted_u_b = dt['encrypted_u_b']
        encrypted_u = encrypted_u_b + dt['encrypted_u_a']
        encrypted_dJ_a = self.compute_encrypted_dJ_a(encrypted_u)
        mask = np.random.rand(len(encrypted_dJ_a))
        encrypted_masked_dJ_a = encrypted_dJ_a + mask
        dt.update({"mask":mask})
        data_to_C = {'encrypted_masked_dJ_a': encrypted_masked_dJ_a}
        self.send_data(data_to_C, self.other_client[client_C_name])
    #A:step6
    def task_3(self):
        print('Astep6')
        dt = self.data
        assert "masked_dJ_a" in dt.keys(), "Error: 'masked_dJ_a' from C in step 2 not successfully received."
        masked_dJ_a = dt['masked_dJ_a']
        dJ_a = masked_dJ_a - dt['mask']
        self.update_weight(dJ_a)
        print(f"A weight: {self.weights}")
# 参与方B在训练过程中及提供特征数据，又提供标签数据
class ClientB(Client):
    def __init__(self,X,y,config):
        super().__init__(config)
        self.X = X
        self.y = y
        self.weights = np.random.uniform(-0.1, 0.1, X.shape[1])
        # self.weights = np.zeros(X.shape[1])
        self.data = {}
    def compute_u_b(self):
        z_b = np.dot(self.X, self.weights)
        u_b = 0.25 * z_b - self.y + 0.5
        return z_b,u_b
    def compute_encrypted_dJ_b(self, encrypted_u):
        encrypted_dJ_b = self.X.T.dot(encrypted_u) + self.config['lambda'] * self.weights
        return encrypted_dJ_b

    def update_weight(self, dJ_b):
        self.weights = self.weights - self.config["lr"] * dJ_b / len(self.X)
    ## B: step2
    def task_1(self, client_A_name):
        print('Bstep2')
        try:
            dt = self.data
            print('Bstep2:',len(dt['masked_dJ_a']))
            print('Bstep2:',dt)
            assert "public_key" in dt.keys(), "Error: 'public_key' from C in step 1 not successfully received."
            public_key = dt['public_key']
        except Exception as e:
            print("B step 1 exception: %s" % e)
        try:
            z_b, u_b = self.compute_u_b()
            to_a = dt['masked_dJ_a']
            encrypted_u_b = np.asarray([public_key.encrypt(x) for x in to_a])
            dt.update({"encrypted_u_b": encrypted_u_b})

            dt.update({"z_b": z_b})

        except Exception as e:
            print("Wrong 1 in B: %s" % e)

        data_to_A = {"encrypted_u_b": encrypted_u_b}

        self.send_data(data_to_A, self.other_client[client_A_name])
    #B:step3
    def task_2(self, client_C_name):
        global encrypted_masked_dJ_b, encrypted_loss, encrypted_u_a, dt
        print('Bstep3')
        try:
            dt = self.data
            print('data from A:',len(dt['encrypted_u_a']))
            assert "encrypted_u_a" in dt.keys(), "Error: 'encrypt_u_a' from A in step 1 not successfully received."
            encrypted_u_a = dt['encrypted_u_a']
            dt.update({"encrypted_u_a":encrypted_u_a})
        except Exception as e:
            print("B step 2 exception: %s" % e)
        data_to_C = {"encrypted_u_a": encrypted_u_a}
        #将A的数据发送给C
        self.send_data(data_to_C, self.other_client[client_C_name])
    #B:step6
    def task_3(self):
        print('Bstep6')
        try:
            dt = self.data
            assert "masked_dJ_b" in dt.keys(), "Error: 'masked_dJ_b' from C in step 2 not successfully received."
            masked_dJ_b = dt['masked_dJ_b']
            dJ_b = masked_dJ_b - dt['mask']
            self.update_weight(dJ_b)
        except Exception as e:
            print("A step 3 exception: %s" % e)
        print(f"B weight: {self.weights}")
# 参与方C在整个训练过程中主要的作用是分发密钥，以及最后的对A和B加密梯度的解密
class ClientC(Client):
    def __init__(self, A_d_shape, B_d_shape, config):
        super().__init__(config)
        self.A_data_shape = A_d_shape
        self.B_data_shape = B_d_shape
        self.public_key = None
        self.private_key = None
        # 保存训练中的损失值
        self.loss = []
    # C:step1
    def task_1(self, client_A_name, client_B_name):
        print('Cstep1')
        try:
            public_key, private_key = paillier.generate_paillier_keypair()
            self.public_key = public_key
            self.private_key = private_key
        except Exception as e:
            print("C step 1 error 1: %s" % e)
        data_to_AB = {"public_key":public_key}
        self.send_data(data_to_AB, self.other_client[client_A_name])
        self.send_data(data_to_AB, self.other_client[client_B_name])
        return
    # C:step5
    def task_2(self, client_B_name):
        global masked_dJ_a, masked_dJ_b
        print('Cstep5')
        try:
            dt = self.data
            print('dtC:',len(dt['encrypted_u_a']))
            assert "encrypted_u_a" in dt.keys() , \
                "Error: 'masked_dJ_a' from A or 'masked_dJ_b' from B in step 2 not successfully received."
            encrypted_u_a = dt['encrypted_u_a']
            print('encrypted_u_a:',encrypted_u_a.shape)
            masked_dJ_a = np.asarray([self.private_key.decrypt(x) for x in encrypted_u_a])
        except Exception as e:
            print("C step 2 exception: %s" % e)
            print("C step 2 exception: %s" % e)
        data_to_B = {"masked_dJ_a": masked_dJ_a}
        self.send_data(data_to_B, self.other_client[client_B_name])
        print('data_to_B',len(data_to_B['masked_dJ_a']))
        print('data_to_B', data_to_B)
        return

# 模拟数据的生成，基于sklearn中的乳腺癌数据集生成模拟数据，参与方A获得部分特征数据，参与方B获得部分特征数据与标签数据
def load_data():
    breast = load_breast_cancer()
    X_train,X_test,y_train,y_test = train_test_split(breast.data,breast.target,random_state=1)
    std = StandardScaler()
    X_train = std.fit_transform(X_train)
    X_test = std.transform(X_test)
    return X_train,y_train,X_test,y_test
def vertically_partition_data(X,X_test,A_idx,B_idx):
    XA = X[:,A_idx]
    XB = X[:,B_idx]
    XB = np.c_[np.ones(X.shape[0]),XB]
    XA_test = X_test[:,A_idx]
    XB_test = X_test[:,B_idx]
    XB_test = np.c_[np.ones(XB_test.shape[0]),XB_test]
    return XA,XB,XA_test,XB_test
def vertical_logistic_regression(X,y,X_test,y_test,config):
    XA, XB, XA_test, XB_test = vertically_partition_data(X, X_test, config['A_idx'], config['B_idx'])
    print('XA:', XA.shape, '   XB:', XB.shape)
    # 各参与方的初始化
    client_A = ClientA(XA,config)
    print("Client_A successfully initialized.")
    client_B = ClientB(XB,y, config)
    print("Client_B successfully initialized.")
    client_C = ClientC(XA.shape,XB.shape,config)
    print("Client_C successfully initialized.")
    # 各参与方之间连接的建立
    client_A.connect("B", client_B)
    client_A.connect("C", client_C)
    client_B.connect("A", client_A)
    client_B.connect("C", client_C)
    client_C.connect("A", client_A)
    client_C.connect("B", client_B)
    # 训练
    for i in range(config['n_iter']):
        #分发公钥
        client_C.task_1("A", "B")
        #A将自己的数据传给B
        client_A.task_1("B")
        #B在获取到A的数据之后将数据传给C，要去掉step3
        client_B.task_2("C")
        #C将B传给自己的A的数据进行解密，然后再传给B
        client_C.task_2("B")
        #B不再按照特定规则传送给A数据，而是将攻击参数传给A，让A按照B想要的去更新参数，再传回给B

        # 已经拿到A传给B的参数了，能不能直接用？
        # 再传回给A，A再更新，需要再从A得到什么？
        # 参数α如何确定？？
        #在给c发送的过程中只有B发送了，c会不会检测是不是不行？
        client_B.task_1("A")

        #client_A.task_2("C")
        #参数更新
        # client_A.task_3()
        # client_B.task_3()
    print("All process done.")
    return True


config = {
    'n_iter': 10,

    'lambda': 10,

    'lr': 0.05,

    'A_idx': [10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29],

    'B_idx': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
}
X, y, X_test, y_test = load_data()
vertical_logistic_regression(X, y, X_test, y_test, config)
