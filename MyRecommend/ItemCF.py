import math
import random
from operator import itemgetter

from MyRecommend.IRecommend import IRecommend


class ItemBasedCF(IRecommend):
    # 初始化相关参数
    def __init__(self, train_set):
        # 找到与目标用户兴趣相似的20部电影，为其推荐10部电影
        self.n_sim_movie = 20
        self.n_rec_movie = 10

        # 将数据集划分为训练集和测试集
        self.trainSet = train_set

        # 相似度矩阵
        self.movie_sim_matrix = {}

        print('Similar movie number = %d' % self.n_sim_movie)
        print('Recommended movie number = %d' % self.n_rec_movie)

    # 计算用户之间的相似度
    def train(self):
        # 构建“电影-用户”倒排索引
        print('Building movie_popular table ...')
        movie_popular: dict[int, int] = {}
        for user, movies in self.trainSet.items():
            for movie in movies:
                movie_popular.setdefault(movie, 0)
                movie_popular[movie] += 1
        print('Build movie_popular table success!')

        print('Build co-rated movies matrix ...')
        for user, movies in self.trainSet.items():
            # 遍历该用户每件物品项
            for m1 in movies:
                # 遍历该用户每件物品项
                for m2 in movies:
                    # 若该项为当前物品，跳过
                    if m1 == m2:
                        continue
                    self.movie_sim_matrix.setdefault(m1, {})
                    self.movie_sim_matrix[m1].setdefault(m2, 0)
                    # 同一个用户，遍历到其他用品则加1
                    self.movie_sim_matrix[m1][m2] += 1
        print('Build co-rated movies matrix success!')

        # 计算电影之间的相似性
        print("Calculating movie similarity matrix ...")
        for m1, related_movies in self.movie_sim_matrix.items():
            for m2, count in related_movies.items():
                # 注意0向量的处理，即某电影的用户数为0
                if movie_popular[m1] == 0 or movie_popular[m2] == 0:
                    self.movie_sim_matrix[m1][m2] = 0
                else:
                    self.movie_sim_matrix[m1][m2] = count / math.sqrt(movie_popular[m1] * movie_popular[m2])
        print('Calculate movie similarity matrix success!')

    # 针对目标用户U，找到其最相似的K个用户，产生N个推荐
    def recommend(self, user):
        k = self.n_sim_movie
        n = self.n_rec_movie
        rank = {}
        watched_movies = self.trainSet[user]

        for movie, rating in watched_movies.items():
            # 遍历与物品item最相似的前k个产品，获得这些物品及相似分数
            for related_movie, w in sorted(self.movie_sim_matrix[movie].items(), key=itemgetter(1), reverse=True)[:k]:
                # 若该物品为当前物品，跳过
                if related_movie in watched_movies:
                    continue
                # 计算用户user对related_movie的偏好值，初始化该值为0
                rank.setdefault(related_movie, 0)
                # 通过与其相似物品对物品related_movie的偏好值相乘并相加。
                # 排名的依据—— > 推荐电影与该已看电影的相似度(累计) * 用户对已看电影的评分
                rank[related_movie] += w * float(rating)
        return sorted(rank.items(), key=itemgetter(1), reverse=True)[:n]


if __name__ == '__main__':
    pass
