import numpy
import numpy as np
from PIL import Image
import cv2
from matplotlib import style
import torch
import random
import copy

style.use("ggplot")


class Tetris:
    piece_colors = [
        (0, 0, 0),
        (255, 255, 0),
        (147, 88, 254),
        (54, 175, 144),
        (255, 0, 0),
        (102, 217, 238),
        (254, 151, 32),
        (0, 0, 255)
    ]

    pieces = [
        [[1, 1],
         [1, 1]],

        [[0, 2, 0],
         [2, 2, 2]],

        [[0, 3, 3],
         [3, 3, 0]],

        [[4, 4, 0],
         [0, 4, 4]],

        [[5, 5, 5, 5]],

        [[0, 0, 6],
         [6, 6, 6]],

        [[7, 0, 0],
         [7, 7, 7]]
    ]

    def __init__(self, height=20, width=10, block_size=20):
        self.height = height
        self.width = width
        self.block_size = block_size
        self.extra_board = np.ones((self.height * self.block_size, self.width * int(self.block_size / 2), 3),
                                   dtype=np.uint8) * np.array([204, 204, 255], dtype=np.uint8)
        self.text_color = (200, 20, 220)
        self.reset()

    def reset(self):
        # 如果一格为0，就是空的
        self.board = [[0] * self.width for _ in range(self.height)]
        self.score = 0
        self.tetrominoes = 0
        self.cleared_lines = 0
        self.bag = list(range(len(self.pieces)))
        random.shuffle(self.bag)

        # 0-6中的一个数字，用于随机生成一个block
        self.ind = self.bag.pop()

        # 6个中的一个形状
        self.piece = [row[:] for row in self.pieces[self.ind]]
        # 方块出现在正中央的上方
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2, "y": 0}
        self.gameover = False
        return self.get_state_properties(self.board)

    def rotate(self, piece):
        '''

        顺时针旋转
        '''
        num_rows_orig = num_cols_new = len(piece)
        # 宽高互换
        num_rows_new = len(piece[0])
        rotated_array = []

        for i in range(num_rows_new):
            new_row = [0] * num_cols_new
            for j in range(num_cols_new):
                new_row[j] = piece[(num_rows_orig - 1) - j][i]
            rotated_array.append(new_row)
        return rotated_array

    def get_state_properties(self, board, extra2=None, col_diff=False):
        """
        :extra2: 二位列表，适配于get_next_state_img
        :return: 组合list [清除掉的行数，总共的holes数量，颠簸度，总block高度]
        """
        lines_cleared, board = self.check_cleared_rows(board)
        holes = self.get_holes(board)
        bumpiness, height = self.get_bumpiness_and_height(board)
        result = [lines_cleared, holes, bumpiness, height]
        if col_diff:
            board = np.array(board)
            mask = board != 0
            invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
            heights = self.height - invert_heights
            for i in range(self.width - 1):
                result.append(abs(heights[i] - heights[i + 1]))

        if extra2:
            result = []
            for row in extra2:
                for item in row:
                    result.append(item)
        return torch.FloatTensor(result)

    def get_holes(self, board):
        """
        如果一列从上到下是:0 0 0 1 1 1 0 1 0 1，那么hole就是2.这里返回所有hole的总数
        """
        num_holes = 0
        for col in zip(*board):
            row = 0
            while row < self.height and col[row] == 0:
                row += 1
            num_holes += len([x for x in col[row + 1:] if x == 0])
        return num_holes

    def get_bumpiness_and_height(self, board):
        """
        返回颠簸度和总block高度
        """
        board = np.array(board)
        mask = board != 0
        # 从上往下数，直到block的最高点的 距离。如果宽为10， 那么形式为[20 17 18 20 20 20 20 20 20 20]
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        # 每一列的最高值。最开始是0
        heights = self.height - invert_heights
        total_height = np.sum(heights)
        # 选取第2-20， 1-19列，做差并求得……颠簸度？
        # [0 3 2 0 0 0 0 0 0]
        currs = heights[:-1]
        # [3 2 0 0 0 0 0 0 0]
        nexts = heights[1:]
        # [3 1 2 0 0 0 0 0 0]
        diffs = np.abs(currs - nexts)
        total_bumpiness = np.sum(diffs)
        return total_bumpiness, total_height

    def get_next_states(self):
        """
        返回的每个state都是block已经下落到底端以后的state
        :return: dict，其中每个元素是 (x轴坐标，rotate次数)，[清除掉的行数，总共的holes数量，颠簸度，总block高度]
        """
        states = {}
        # 选出一块piece
        piece_id = self.ind
        # piece的形状
        curr_piece = [row[:] for row in self.piece]

        # 计算不同形状有几种rotate的结果
        if piece_id == 0:  # O piece
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:
            num_rotations = 2
        else:
            num_rotations = 4

        # 对于不同的下落形状（num_rotation)
        for i in range(num_rotations):
            valid_xs = self.width - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [row[:] for row in curr_piece]
                # 注意！！！y越大，board上越靠下。所以是从上往下遍历的
                pos = {"x": x, "y": 0}
                while not self.check_collision(piece, pos):
                    pos["y"] += 1
                # ???
                self.truncate(piece, pos)
                board = self.store(piece, pos)
                states[(x, i)] = self.get_state_properties(board)
            curr_piece = self.rotate(curr_piece)
        return states

    def count_num(self, piece):
        id = 0
        for row in piece:
            for item in row:
                if item != 0:
                    id = item - 1
                    break
        if id == 0:  # O piece
            num_rotations = 1
        elif id == 2 or id == 3 or id == 4:
            num_rotations = 2
        else:
            num_rotations = 4
        return num_rotations

    def get_next_state_img(self):
        states = {}
        piece_id = self.ind
        curr_piece = [row[:] for row in self.piece]

        if piece_id == 0:  # O piece
            num_rotations = 1
        elif piece_id == 2 or piece_id == 3 or piece_id == 4:
            num_rotations = 2
        else:
            num_rotations = 4

        for i in range(num_rotations):
            valid_xs = self.width - len(curr_piece[0])
            for x in range(valid_xs + 1):
                piece = [row[:] for row in curr_piece]
                pos = {"x": x, "y": 0}
                while not self.check_collision(piece, pos):
                    pos["y"] += 1
                self.truncate(piece, pos)
                temp_board = copy.deepcopy(self.board)
                for row in range(self.height):
                    for item in range(self.width):
                        if temp_board[row][item] != 0:
                            temp_board[row][item] = 255

                board = self.store(piece, pos)
                temp_board = self.store_img(piece, pos, temp_board)
                for row in range(self.height):
                    for item in range(self.width):
                        if not (temp_board[row][item] == 0 or temp_board[row][item] == 255):
                            temp_board[row][item] = 150

                states[(x, i)] = self.get_state_properties(board, temp_board, col_diff=False)
                # for row in temp_board:
                #     for item in row:
                #         #states[(x, i)].append(item)
            curr_piece = self.rotate(curr_piece)
        return states

    def get_current_board_state(self):
        '''

        board + 当前位置显示piece
        '''
        board = [x[:] for x in self.board]
        for y in range(len(self.piece)):
            for x in range(len(self.piece[y])):
                board[y + self.current_pos["y"]][x + self.current_pos["x"]] = self.piece[y][x]
        return board

    def new_piece(self):
        '''

        生成一个新的block，把current_pos设为：屏幕中央-block的一半
        '''
        if not len(self.bag):
            self.bag = list(range(len(self.pieces)))
            random.shuffle(self.bag)
        self.ind = self.bag.pop()
        self.piece = [row[:] for row in self.pieces[self.ind]]
        self.current_pos = {"x": self.width // 2 - len(self.piece[0]) // 2,
                            "y": 0
                            }
        if self.check_collision(self.piece, self.current_pos):
            self.gameover = True

    def check_collision(self, piece, pos):
        '''

        检查是否发生碰撞。如果低于最低高度或者刚好不碰撞：返回True
        '''
        future_y = pos["y"] + 1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if future_y + y > self.height - 1 or self.board[future_y + y][pos["x"] + x] and piece[y][x]:
                    return True
        return False

    def truncate(self, piece, pos):
        '''

        用来判断加上block之后会不会导致gameover
        '''
        gameover = False
        last_collision_row = -1
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x]:
                    if y > last_collision_row:
                        last_collision_row = y

        if pos["y"] - (len(piece) - last_collision_row) < 0 and last_collision_row > -1:
            while last_collision_row >= 0 and len(piece) > 1:
                gameover = True
                last_collision_row = -1
                del piece[0]
                for y in range(len(piece)):
                    for x in range(len(piece[y])):
                        if self.board[pos["y"] + y][pos["x"] + x] and piece[y][x] and y > last_collision_row:
                            last_collision_row = y
        return gameover

    def store(self, piece, pos):
        '''

        储存添加上block之后的board
        '''
        board = [x[:] for x in self.board]
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not board[y + pos["y"]][x + pos["x"]]:
                    board[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return board

    def store_img(selfself, piece, pos, img):
        for y in range(len(piece)):
            for x in range(len(piece[y])):
                if piece[y][x] and not img[y + pos["y"]][x + pos["x"]]:
                    img[y + pos["y"]][x + pos["x"]] = piece[y][x]
        return img

    def check_cleared_rows(self, board):
        '''
        遍历每一行，如果一行中没有0，就（调用remove_row）把这行去掉

        :return: （删掉了几行，删完后的board）
        '''
        to_delete = []
        for i, row in enumerate(board[::-1]):
            if 0 not in row:
                to_delete.append(len(board) - 1 - i)
        if len(to_delete) > 0:
            board = self.remove_row(board, to_delete)
        return len(to_delete), board

    #
    def remove_row(self, board, indices):
        '''

        去掉indices所在的行，并在总体上面加一个空行

        :return: 删掉后的board
        '''
        for i in indices[::-1]:
            del board[i]
            board = [[0 for _ in range(self.width)]] + board
        return board

    def step(self, action, render=True, video=None, name="Deep Q-Learning Tetris"):
        '''
        传递(x轴pos，旋转次数), step会更新board

        要更换界面名称，需要传递name参数
        :return: 返回score, 是否gameover。
        '''
        x, num_rotations = action
        # set位置
        self.current_pos = {"x": x, "y": 0}
        # 旋转指定次数
        for _ in range(num_rotations):
            self.piece = self.rotate(self.piece)
        # 不断将position往下探
        while not self.check_collision(self.piece, self.current_pos):
            self.current_pos["y"] += 1
            # 是否图形化界面
            if render:
                self.render(video, name)
        # 查看是否溢出屏幕，是则gameover
        overflow = self.truncate(self.piece, self.current_pos)
        if overflow:
            self.gameover = True
        # 把piece放在current_pos上
        # current_pos是piece的左上角的坐标
        self.board = self.store(self.piece, self.current_pos)

        # 删掉了几行，删掉后的board（已经把删除行以上的部分全部往下平移过）
        lines_cleared, self.board = self.check_cleared_rows(self.board)
        score = 1 + (lines_cleared ** 2) * self.width
        self.score += score
        # 每加一块piece，就加一个tetrominoes
        self.tetrominoes += 1
        self.cleared_lines += lines_cleared
        if not self.gameover:
            self.new_piece()
        if self.gameover:
            self.score -= 2

        return score, self.gameover

    def render(self, video=None, name="Deep Q-Learning Tetris"):
        if not self.gameover:
            # 给每个像素上色。颜色取决于block的shape
            img = [self.piece_colors[p] for row in self.get_current_board_state() for p in row]
        else:
            img = [self.piece_colors[p] for row in self.board for p in row]
        img = np.array(img).reshape((self.height, self.width, 3)).astype(np.uint8)
        img = img[..., ::-1]
        img = Image.fromarray(img, "RGB")

        img = img.resize((self.width * self.block_size, self.height * self.block_size), 0)
        img = np.array(img)
        img[[i * self.block_size for i in range(self.height)], :, :] = 0
        img[:, [i * self.block_size for i in range(self.width)], :] = 0

        img = np.concatenate((img, self.extra_board), axis=1)

        # 在旁边显示score pieces lines的信息
        cv2.putText(img, "Score:", (self.width * self.block_size + int(self.block_size / 2), self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.score),
                    (self.width * self.block_size + int(self.block_size / 2), 2 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Tetrominoes:", (self.width * self.block_size + int(self.block_size / 2), 4 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.tetrominoes),
                    (self.width * self.block_size + int(self.block_size / 2), 5 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        cv2.putText(img, "Lines:", (self.width * self.block_size + int(self.block_size / 2), 7 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)
        cv2.putText(img, str(self.cleared_lines),
                    (self.width * self.block_size + int(self.block_size / 2), 8 * self.block_size),
                    fontFace=cv2.FONT_HERSHEY_DUPLEX, fontScale=1.0, color=self.text_color)

        if video:
            video.write(img)

        cv2.imshow(name, img)
        cv2.waitKey(1)

    def yaq_state(self):
        '''
        # max_x: 当前最高x
        # std_x: 标准差
        # avg_x: 均值
        # max_diff_x: 最大高度差
        '''
        board = np.array(self.board)
        mask = board != 0
        invert_heights = np.where(mask.any(axis=0), np.argmax(mask, axis=0), self.height)
        heights = self.height - invert_heights
        Mean = numpy.mean(heights)
        st_dev = np.std(heights)
        Max = max(heights)
        Min = min(heights)
        max_diff = Max - Min
        return Max, st_dev, Mean, max_diff

    def heuristic_func(self,board):
        hole_num = self.get_holes(board)
        maxHeight = self.yaq_state()[0]
        avgHeight = self.yaq_state()[2]
        diff = self.yaq_state()[3]
        return -maxHeight-2.5*avgHeight-1.25*hole_num-1.25*diff

    def get_best_action(self):
        possible_actions = self.get_next_states().keys()
        best_action = None
        max_h = -99999

        for action in possible_actions:
            current_h = self.heuristic_func(self.board)
            current_h += (-1.25*self.get_next_states()[action][2]-1.25*self.get_next_states()[action][1]+40*self.get_next_states()[action][0]-1.25*self.get_next_states()[action][3])
            if current_h > max_h:
                max_h = current_h
                best_action = action
        return best_action

