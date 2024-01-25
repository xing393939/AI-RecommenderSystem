import random
from MyRecommend import IRecommend
from MyRecommend.ItemCF import ItemBasedCF
from MyRecommend.UserCF import UserBasedCF


class Main:
    # 初始化相关参数
    def __init__(self):
        # 将数据集划分为训练集和测试集
        self.trainSet = {}
        self.testSet = {}
        self.algorithm: IRecommend = None
        self.n_rec_movie = 10
        self.movie_count = 0

    # 读文件，返回文件的每一行
    @staticmethod
    def load_file(filename):
        with open(filename, 'r') as f:
            for i, line in enumerate(f):
                # 去掉文件第一行的title
                if i == 0:
                    continue
                yield line.strip('\r\n')
        print('Load %s success!' % filename)

    # 读文件得到“用户-电影”数据 ,分为测试集和训练集
    def get_dataset(self, filename, pivot=0.75):
        train_set_len = 0
        test_set_len = 0
        movie_set = set()
        for line in self.load_file(filename):
            user, movie, rating, timestamp = line.split(',')
            if random.random() < pivot:
                self.trainSet.setdefault(user, {})
                self.trainSet[user][movie] = rating
                train_set_len += 1
                movie_set.add(movie)
            else:
                self.testSet.setdefault(user, {})
                self.testSet[user][movie] = rating
                test_set_len += 1
        self.movie_count = len(movie_set)
        print('Split trainingSet and testSet success!')
        print('TrainSet = %s' % train_set_len)
        print('TestSet = %s' % test_set_len)
        print('TrainSet movie number = %d' % len(movie_set))

    def algorithm_init(self, alg):
        self.algorithm = alg
        self.algorithm.train()

    # 产生推荐并通过准确率、召回率和覆盖率进行评估
    def evaluate(self):
        print("Evaluation start ...")
        # 准确率和召回率
        hit = 0
        rec_count = 0
        test_count = 0
        # 覆盖率
        all_rec_movies = set()

        for i, user, in enumerate(self.trainSet):
            test_movies = self.testSet.get(user, {})
            rec_movies = self.algorithm.recommend(user)
            for movie, w in rec_movies:
                if movie in test_movies:
                    hit += 1
                all_rec_movies.add(movie)
            rec_count += self.n_rec_movie
            test_count += len(test_movies)

        precision = hit / (1.0 * rec_count)
        recall = hit / (1.0 * test_count)  #
        coverage = len(all_rec_movies) / (1.0 * self.movie_count)
        print('precisioin=%.4f\trecall=%.4f\tcoverage=%.4f' % (precision, recall, coverage))


if __name__ == '__main__':
    rating_file = '../Recall/CollaborativeFiltering/ml-latest-small/ratings.csv'
    main = Main()
    main.get_dataset(rating_file)
    main.algorithm_init(ItemBasedCF(main.trainSet))
    main.evaluate()
    main.algorithm_init(UserBasedCF(main.trainSet))
    main.evaluate()
