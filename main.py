import collections
import json
import os

import numpy as np
import pandas as pd


class AntColony:
    """
    num_ants：此參數決定演算法中使用的螞蟻數量。更多的螞蟻可能會探索更多的路徑，但這也增加了計算成本。
    的良好值num_ants取決於問題的規模和可用的計算資源。
    alpha：此參數控制費洛蒙水準對選擇城市機率的影響。數值越高alpha，意味著螞蟻越有可能選擇信息素水平
    較高的城市（即以前有很多螞蟻訪問過的城市）。這可以幫助演算法更快收斂，但也可能導致過早收斂到次優解。
    beta：此參數控制啟發式資訊（在本例中為距離的倒數）對選擇城市的機率的影響。數值越高，beta代表螞蟻
    越有可能選擇距離目前城市較近的城市。這可以幫助演算法找到更短的路徑，但也可能導致螞蟻陷入局部最優。
    evaporation_rate：此參數決定費洛蒙水平隨時間蒸發（或衰減）的速率。數值越高evaporation_rate意
    味著信息素水平下降得越快，使演算法更容易探索新路徑。然而，如果蒸發率太高，演算法可能會過快地丟失有
    用的信息素信息，導致收斂性差。
    """

    def __init__(
        self,
        graph,
        num_ants,
        num_iterations,
        alpha=1.5,
        beta=3.0,
        evaporation_rate=0.3,
    ):
        self.graph = graph
        self.num_ants = num_ants
        self.num_iterations = num_iterations
        self.alpha = alpha
        self.beta = beta
        self.evaporation_rate = evaporation_rate
        # Initialize pheromone levels
        all_cities = set()
        for city, neighbors in graph.items():
            all_cities.add(city)
            all_cities.update(neighbors.keys())

        self.pheromones = {
            city: {neighbor: 1.0 for neighbor in all_cities} for city in all_cities
        }

    """
    此方法初始化螞蟻以進行新的迭代。
    它創建num_ants螞蟻，每隻螞蟻都有一個包含 的列表start_city，
    距離為 0.0，以及一個包含start_city作為訪問城市的集合。
    """

    def _initialize_ants(self, start_city):
        self.ants = []
        for _ in range(self.num_ants):
            ant = {"path": [start_city], "distance": 0.0, "visited": {start_city}}
            self.ants.append(ant)

    """
    該方法根據信息素水平和啟發式資訊為給定的螞蟻選擇下一個城市。
    它根據信息素水平和啟發值（距離的倒數）計算選擇每個未訪問城市的機率。
    然後它根據這些機率隨機選擇一個城市。
    如果沒有未訪問過的城市，則返回None。
    """

    def _select_next_city(self, ant):
        current_city = ant["path"][-1]
        unvisited_cities = [
            city for city in self.graph[current_city] if city not in ant["visited"]
        ]

        if not unvisited_cities:
            return None

        probabilities = []
        pheromones = self.pheromones[current_city]
        for city in unvisited_cities:
            pheromone = pheromones[city] ** self.alpha
            """
            我們透過取當前城市與未造訪過的城市之間的距離的倒數，
            並將其計算為 的冪次方來計算啟發式貢獻self.beta。
            該值表示啟發式資訊（在本例中為距離的倒數）對選擇城市的機率的影響。
            """
            heuristic = (1.0 / self.graph[current_city][city]) ** self.beta
            probabilities.append(pheromone * heuristic)

        probabilities = np.array(probabilities)
        probabilities /= probabilities.sum()

        return np.random.choice(unvisited_cities, p=probabilities)

    """
    此方法根據螞蟻所走的路徑更新信息素水平。
    首先，它將現有的費洛蒙水平乘以 來蒸發(1.0 - evaporation_rate)。
    然後，對於每隻螞蟻，它會以與 成比例的量增加螞蟻訪問過的邊緣上的信息素水平1.0 / ant["distance"]。
    """

    def _update_pheromones(self):
        for city, neighbors in self.pheromones.items():
            for neighbor in neighbors:
                self.pheromones[city][neighbor] *= 1.0 - self.evaporation_rate

        for ant in self.ants:
            for i in range(len(ant["path"]) - 1):
                city = ant["path"][i]
                next_city = ant["path"][i + 1]
                self.pheromones[city][next_city] += 1.0 / ant["distance"]
                self.pheromones[next_city][city] += 1.0 / ant["distance"]

    """
    此方法為每隻螞蟻建立解決方案（旅行）。
    它迭代螞蟻並重複選擇下一個城市，_select_next_city直到所有城市都被訪問過。
    start_city訪問所有城市後，它會透過將 添加到螞蟻路徑的末端並相應地更新距離來結束遊覽。
    """

    def _construct_solutions(self, start_city):
        for ant in self.ants:
            while len(ant["path"]) < len(self.graph):
                next_city = self._select_next_city(ant)
                if next_city is None:
                    break
                ant["path"].append(next_city)
                ant["visited"].add(next_city)
                ant["distance"] += self.graph[ant["path"][-2]][next_city]

            # Closing the tour by returning to the start city
            ant["path"].append(start_city)
            ant["distance"] += self.graph[ant["path"][-2]][start_city]

    """
    這是使用蟻群最佳化演算法解決旅行商問題（TSP）的主要方法。
    它迭代num_iterations多次。
    在每次迭代中，它都會初始化螞蟻、建立解決方案並更新信息素水平。
    它追蹤最佳路徑（遊覽）及其距離。
    經過所有迭代後，它會返回最佳路徑及其距離。
    """

    def solve(self, start_city):
        best_path = None
        best_distance = float("inf")
        for _ in range(self.num_iterations):
            self._initialize_ants(start_city)
            self._construct_solutions(start_city)
            self._update_pheromones()
            for ant in self.ants:
                if ant["path"][-1] == start_city and ant["distance"] < best_distance:
                    best_distance = ant["distance"]
                    best_path = ant["path"]
        return " -> ".join(best_path), best_distance


def get_nodes():
    filepath = f"./citys.json"
    if not os.path.isfile(filepath):
        df = pd.read_html(
            "https://www.discovery-campervans.com.au/distance_guide.php", index_col=0
        )[0]
        nodes = []
        for i, row in enumerate(df.iterrows()):
            row_data = []
            for col in df.columns:
                if row[1][col] != row[1][col]:  # Check for NaN values
                    continue
                else:
                    row_data.append([df.index[i], col, row[1][col]])
            nodes.extend(row_data)
        with open("citys.json", "w") as f:
            f.write(json.dumps(nodes))
    else:
        with open("citys.json", "r") as f:
            nodes = json.load(f)
    return nodes


def create_graph(nodes):
    graph = collections.defaultdict(dict)
    for node1, node2, distance in nodes:
        graph[node1][node2] = distance
        graph[node2][node1] = distance  # Assuming undirected graph
    return graph


def main():
    nodes = get_nodes()
    graph = create_graph(nodes)
    ant_colony = AntColony(graph, num_ants=10, num_iterations=100)

    start_city = "Brisbane"
    best_path, best_distance = ant_colony.solve(start_city)
    return f"{best_path}, distance: {best_distance} km"


if __name__ == "__main__":
    answer = main()
    print(answer)
