import math
from operator import itemgetter

from MyRecommend.IRecommend import IRecommend


class UserBasedCF(IRecommend):
    # 初始化相关参数
    def __init__(self, train_set):
        # 找到与目标用户兴趣相似的20个用户，为其推荐10部电影
        self.n_sim_user = 20
        self.n_rec_movie = 10

        # 将数据集划分为训练集和测试集
        self.trainSet = train_set

        # 用户相似度矩阵
        self.user_sim_matrix = {}

        print('Similar user number = %d' % self.n_sim_user)
        print('Recommended movie number = %d' % self.n_rec_movie)

    # 计算用户之间的相似度
    def train(self):
        # 构建“电影-用户”倒排索引
        print('Building movie-user table ...')
        movie_user: dict[int, set] = {}
        for user, movies in self.trainSet.items():
            for movie in movies:
                movie_user.setdefault(movie, set())
                movie_user[movie].add(user)
        print('Build movie-user table success!')

        print('Build user co-rated movies matrix ...')
        for movie, users in movie_user.items():
            for u in users:
                for v in users:
                    if u == v:
                        continue
                    self.user_sim_matrix.setdefault(u, {})
                    self.user_sim_matrix[u].setdefault(v, 0)
                    self.user_sim_matrix[u][v] += 1
        print('Build user co-rated movies matrix success!')

        # 计算用户之间的余弦相似度
        print('Calculating user similarity matrix ...')
        for u, related_users in self.user_sim_matrix.items():
            for v, count in related_users.items():
                self.user_sim_matrix[u][v] = count / math.sqrt(len(self.trainSet[u]) * len(self.trainSet[v]))
        print('Calculate user similarity matrix success!')

    # 针对目标用户U，找到其最相似的K个用户，产生N个推荐
    def recommend(self, user):
        k = self.n_sim_user
        n = self.n_rec_movie
        rank = {}
        watched_movies = self.trainSet[user]

        for v, wuv in sorted(self.user_sim_matrix[user].items(), key=itemgetter(1), reverse=True)[0:k]:
            for movie in self.trainSet[v]:
                if movie in watched_movies:
                    continue
                rank.setdefault(movie, 0)
                rank[movie] += wuv
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[0:n]


if __name__ == '__main__':
    pass
