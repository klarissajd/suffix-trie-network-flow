__author__ = "Klarissa Jutivannadevi"
__student_id__ = "32266014"

import copy
import math


# Task 1 - Sharing the Meals

class Graph:
    """
    A class designed to create a network flow. This class contains method that are required in getting the
    final network flow. In this graph, Ford-Fulkerson method is used to get the maximum flow and the network
    flow to get the maximum flow and also a modification of circulation graph to allow the lower bound constraint
    """

    def __init__(self, availability):
        """
        A constructor of the Graph class containing the useful variable which are mostly generated from the input
        :Input:
            :param availability: A list of lists indicating days and within is each person's available time
        :Attributes:
        days: the total days to be allocated for breakfast and dinner
        base_nodes: the nodes that will always exist (containing source, 5 people, and target)
        nodes_create: the total nodes after the nodes for days are included
        copied: The initial nodes when the ford fulkerson algorithm is not implemented (unchanged)
        returned: a list containing 2 lists which is the initial graph and final graph after ford fulkerson is applied
        """
        self.days = len(availability)
        self.base_nodes = 7  # source + 5 people + target
        self.nodes_create = self.base_nodes + self.days * 3  # days * 3 = breakfast, dinner, either
        self.nodes = [[0] * self.nodes_create for _ in
                      range(self.nodes_create)]  # ASC: O(n^2) where n is days and nodes createed are 3n + 7
        self.base_add_edges()
        self.allocate_flow(availability)
        self.copied = copy.deepcopy(self.nodes)
        self.returned = [self.copied, self.circulation_demand()]

    # REFERENCE:
    # index 0 : source
    # index 1 - 5 : people 0-4
    # index 6: target
    # the rest of the index : nodes based on its day
    def base_add_edges(self):
        """
        This method is a method that is used to create just the default node. This means that
        it only add edge from source to each person and from breakfast and dinner to target. Since
        it is the base, the edge added from the source to each person will only be the lower bound.
        This is done to simplify when creating network flow with circular demand (to simplify, demand
        is not added and instead source and target are created immediately).
        :return: A graph with edges from source to person (minimum flow) and breakfast and dinner to
        target (each with flow of 1)
        :Time complexity: O(n) where n is the length of the nodes existed for the graph (to be exact,
        3n + 7 where 7 is the required starting nodes and 3n is for (either + breakfast + dinner) * 3
        :Aux space complexity: O(1) since no new list by n length are created, only updating the value
        within the adjacency matrix
        """
        lower_bound = math.floor(self.days * 0.36)

        # TC: O(1) where looping only done 5 times for any input (since people are constant, which is 5)
        for i in range(1, 6):
            self.nodes[0][i] = lower_bound  # TC: O(1) only changing value

        # update the flow from of either node to breakfast and dinner (pattern: breakfast, dinner, either)
        # TC: O(n) where it loops through the input n starting from certain index in nodes (indicating schedule) and
        # skips by the interval of 3 each time (since pattern of 3 is a set)
        for i in range(7, len(self.nodes), 3):
            # "either" needs to have a flow to both breakfast and dinner
            self.nodes[i + 2][i] = 1
            self.nodes[i + 2][i + 1] = 1

        # TC: O(n) where it loops through the schedule to update flow breakfast and dinner to target
        for i in range(7, len(self.nodes), 3):
            self.nodes[i][6] = 1
            self.nodes[i + 1][6] = 1

    # flow from individual people to schedule
    def allocate_flow(self, availability):
        """
        This method is used to add edges with flow of 1 based on the availability of the
        person. In this function, each person will have an edge to any of either, breakfast,
        or dinner as long as it is their available time.
        :Input:
            :param availability: A list of lists indicating what days and schedule each person
            are available at
        :return: Instead of returning, it modifies the node variable in the class to create
        a complete graph
        :Time complexity: O(n) where n is the number of days that needs to be allocated
        :Aux space complexity: O(1) where only an update is needed for an element
        """
        for days in range(len(availability)):  # TC: O(n) where n is number of days
            for person in range(len(availability[days])):  # TC: O(1) since there is only 5 people
                # if they are available (indicated by element being 1-3)
                if availability[days][person] != 0:
                    # add an edge with flow of 1
                    self.nodes[person + 1][days * 3 + availability[days][person] + 6] = 1

    def bfs(self, source, target, parent):
        """
        The breadth-first-search is a helper function in order to execute ford-fulkerson
        algorithm. Like a usual bfs, this method just simply finds a path from the starting
        node (the source) to the desired node (the target).
        :Input:
            :param source: The starting node to be traversed
            :param target: The reaching node to indicate bfs succeeds
            :param parent: A list containing the parent nodes of the traversal. (-1 on the index
            means that that index is the root (a.k.a starting)
        :return: Returns boolean to indicate that there is a path from source to target
        (indicating flow can go)
        :Time complexity: O(n^2) where it iterates through n * n due to a adjacency matrix
        being created
        :Aux space complexity: O(n) where visited list is created to indicate whether node has
        been visited
        """
        visited = [False] * self.nodes_create
        # Create a queue for BFS
        q = [source]
        visited[source] = True
        while q:
            u = q.pop(0)
            for i in range(len(self.nodes[u])):
                if visited[i] is False and self.nodes[u][i] > 0:
                    q.append(i)
                    visited[i] = True
                    parent[i] = u
                    if i == target:
                        return True

        return False

    # Returns the maximum flow from s to t in the given graph
    def ford_fulkerson(self, source, target):
        """
        Reference: https://www.geeksforgeeks.org/ford-fulkerson-algorithm-for-maximum-flow-problem/
        This method is used in order to find the maximum flow that a graph can do. This function
        implement bfs in order to find the optimal solution, where each time it finds a path that
        allows more flow, it recomputed the edge from certain nodes and will eventually stop once
        there is no longer a path giving more flow to the target.
        :Input:
            :param source: The starting flow
            :param target: The reaching flow point
        :return: A revised graph where the flow has been allocated to edge to achieve a maximum flow
        :Time complexity: O(n^3) where O(n^2) comes from the bfs and is iterated using while loop
        (worst case bfs will search all options which is n). Since flow = 1 (based on the requirement),
        it can be ignored
        :Aux space complexity: O(n) where parent node is created to keep track of the preceding nodes.
        The rest of the computation does not involve creating new list
        """
        parent = [-1] * self.nodes_create       # ASC: O(n) where new list is created to store parent node
        max_flow = 0  # There is no flow initially

        while self.bfs(source, target, parent):     # TC: O(n^3) from bfs itself and the additional while condition
            flow = math.inf
            current = target
            # when position has not reached source
            while current != source:
                # take the existing flow (if there is in for that edge)
                flow = min(flow, self.nodes[parent[current]][current])
                # update for the next iteration they will now be the current (move back)
                current = parent[current]

            # Add path flow to overall flow
            max_flow += flow

            t = target
            while t != source:      # TC: O(n) in case it traverse to all before reaching source
                u = parent[t]
                # when flow is transferred, change the weight
                self.nodes[u][t] -= flow
                self.nodes[t][u] += flow
                # update the position to reach source
                t = parent[t]

        return self.nodes, max_flow

    def circulation_demand(self):
        """
        This is a modified version of the supposed circulation demand graph where the source and
        target nodes has been created and only modifying edge is done. Instead of creating the graph
        at once, it is separated because I want to ensure that the lower bound has been fulfilled for
        all people, and the rest can just be allocated to any people as long as it is withing the
        capacity. This method only checks whether all the demand from source are transferred to the
        target
        :return: The updated graph if all demands are transferred or else none
        :Time complexity: O(n^3) where ford_fulkerson is done in order to find the graph to fulfill the
        lower bound
        :Aux space complexity: O(n) which is similar to ford_fulkerson method as it takes that method
        to get the updated graph.
        """
        lower_bound = math.floor(self.days * 0.36)
        # getting the new graph and the max_flow
        graph, max_flow = self.ford_fulkerson(0, 6)
        if max_flow != lower_bound * 5:     # max_flow should be equal to lower bound since all person need to meet this
            return None             # if it does not meet, it's an early indication showing that it's not feasible
        else:
            final_network = self.add_lower_bound()      # if it is, add extra flow (explained in add_lower_bound())
            return final_network

    def add_lower_bound(self):
        """
        A method used to add the excess flow that is not previously included (in order to meet the
        constraints of the lower bound) to the graph. It will then repeat the same method using
        fold_fulkerson to find the additional flows that can be received and will return the
        graph including the maximum flow.
        :return: The updated graph (after the leftover flow is added) and the new maximum_flow
        achieved by doing the fold_fulkerson method
        :Time complexity: O(n^3) same with ford_fulkerson method since that is the biggest
        complexity and is called in this method
        :Aux space complexity: O(n) with the same reason. The method excluding the ford_fulkerson
        method itself contains O(1) since no new lists created
        """
        lower_bound = math.floor(self.days * 0.36)
        upper_bound = math.ceil(self.days * 0.44)
        for i in range(1, 6):               # add flow with edge from source to people with the possible leftover
            self.nodes[0][i] = upper_bound - lower_bound
        new_g = self.ford_fulkerson(0, 6)   # returning a tuple with graph and max flow
        return new_g


def allocate(availability):
    """
    The function which will put the results of the graph and reformat the result to a
    readable schedule with list of breakfast and dinner together with the person who
    are allocated each day.
    :Input:
        :param availability: The available schedule of each person given in a form
        of list of lists with each outer list indicating day and inner list indicate
        schedule available for every person index
    :return: A tuple giving (breakfast, dinner) where both are a list
    :Time complexity: O(n^2) with the same reason of ford_fulkerson() being implemented
    when the graph object is used
    :Aux space complexity: O(n) where the complexity comes from creating a breakfast list
    ana dinner list
    """
    graph = Graph(availability)
    old_graph = graph.returned[0]           # the initial graph before any flow is moved
    new_graph = graph.returned[1][0]        # returned[1] gives a tuple of graph and max_flow
    # in case the graph does not satisfy the requirements from the start
    if new_graph is None:
        return None
    breakfast = [None] * graph.days         # ASC: O(n) where n is days
    dinner = [None] * graph.days
    tuples = (breakfast, dinner)
    temp_holder = []                    # ASC: O(n) in case all person are available either on each day

    for i in range(1, 6):  # TC: O(1) just check the changes of flow in these 5 people
        for j in range(7, graph.nodes_create):  # TC: O(n) where it iterates to all the nodes to find flow change
            if old_graph[i][j] - new_graph[i][j] == 1:      # indicate flow is transferred
                day = (j - 7) // 3                  # adjusting to find index (since 0-6 is fixed)
                schedule = (j - 7) % 3
                if schedule <= 1:           # since 0 and 1 indicate breakfast and dinner in this case
                    tuples[schedule][day] = i - 1
                else:           # if available on either, cannot be allocated first
                    temp_holder.append([day, i - 1])

    for remain in temp_holder:          # iterate through the list containing flow from either
        # since either can be placed anywhere, just checked the same day with None (meaning it has not been
        # allocated to any person)
        if tuples[0][remain[0]] is None:
            tuples[0][remain[0]] = remain[1]
        elif tuples[1][remain[0]] is None:
            tuples[1][remain[0]] = remain[1]

    counter = 0             # count how many unfilled schedule left

    for i in range(len(tuples)):        # O(1) for breakfast and dinner
        for j in range(len(tuples[i])):     # O(n) where n is days
            if tuples[i][j] is None:        # count None and change to 5 (indicating buy)
                tuples[i][j] = 5
                counter += 1

    # fulfilling the last condition where cannot buy more than the supposed count
    if counter > math.floor(graph.days * 0.1):
        return None
    return tuples


# Task 2  - Similarity Detector

def collect_suffix(string):
    """
    Collect all the suffixes to be added to the trie
    :param string: Input 1 + Input 2 combined
    :return: a list of the suffixes up to the entire string
    :Time complexity: O((N^2+M^2) where an iteration of the entire string is done backward
    from the end in order to get each suffix for an addition of a letter. The letter taken
    is added a character each time + list slicing
    :Space complexity: O(N^2+M^2) where the length of the list returned will be the length reaching
    to its maximum suffix (which is N^2/2 or M^2/2 by mathematical calculation)
    """
    suffixes = []   # SC: O(N^2+M^2) once appended
    # it is iterated backward to collect the total suffixes (since suffix start from the end)
    for i in range(len(string) - 1, -1, -1):
        suffixes.append(string[i:])         # TC: O(N) where N is string (can be N or M)
    return suffixes


class SuffixTrieNode:
    """
    A class that creates node and store information of the nodes that will
    be useful for checking when any computation is done
    :Input: nothing
    :return: A SuffixTrieNode object that create nodes for every starting
    node and the information such as children, parents, etc.
    """
    def __init__(self, individual_char="", chars_collected=[]):
        """
        A constructor method that contains information for every node created
        :param individual_char: The character the node is holding
        :param chars_collected: The parent nodes character accumulated
        :Time complexity: O(1) only assigning variable is done
        :Aux space complexity: O(1) the maximum possible length of list it can
        have (children) is an approximate amount of an existing character.
        """
        self.individual_char = individual_char
        self.children = []  # list of nodes
        self.chars_collected = chars_collected


class SuffixTrie:
    """
    The SuffixTrie class is a class that contains computations in order to create
    a suffix trie. It only contains one method which is an insertion method
    that is used for updating a trie for each suffix.
    :Attributes:
        nodes: A starting node with no information within to be branched out for
        each suffix
    """
    def __init__(self, suffixes):
        """
        This is a constructor method of the class SuffixTrie. It creates an initial
        empty node inside the nodes variable and also process the creation of trie by
        looping through the input lists and calling the insertion suffix
        :Input:
            :param suffixes: The list of suffixes build from a string
        :Time complexity: O(n^2) where n^2 is the suffix is iterated each time and
        insertion_suffix is called to add character to the trie
        :Aux space complexity: O(n^2) where n^2 is the length of the list of the
        input suffixes.
        """
        self.nodes = [SuffixTrieNode()]     # initial node (the root for branching out)
        for suffix in suffixes:             # iterates through the suffix to create a trie for the string by suffix
            self.insertion_suffix(suffix)   # inserting nodes to create entire suffix trie

    def insertion_suffix(self, suffix):
        """
        The insertion_suffix method takes a string and iterate through each character to find
        any existing character and will create a new node below the parent node if it does not
        exist. To indicate it's a child node, it will be appended to the node property of the
        parent node where it contains list of child
        :Input:
            :param suffix: A substring of an input string (essentially is just a suffix)
        :return: does not return anything but modify the class's node
        :Time complexity: O(N^2) where N is the first input string being used for this computation
        :Aux space complexity: O(N^2) since a suffix trie is created, and they consume N^2 compared
        to the initial string
        """
        node = self.nodes[0]            # starting from the root node
        for char in suffix:             # iterate every character from the substring suffix
            match = None
            for child in node.children:     # iterate through the children node of the parent node
                if child.individual_char == char:       # check whether character has existed (using individual_char from Node class)
                    match = child           # if existed, just break the loop and set match to child
                    break

            added_char = node.chars_collected.copy()            # copy the char_collected so that it won't be modified (since it's a list)
            added_char.append(char)                     # update the copied list by appending the new char
            new_node = SuffixTrieNode(char, added_char)     # create a node and inputting the information required

            if match is None:                   # if match is still None it means that there is no same node
                node.children.append(new_node)          # add the node as a child node
                node = new_node                 # update the node as a parent node
            else:
                node = match                # just update the parent node to this node


def second_sentence(trie, suffix, max_string):
    """
    This method is used to create a suffix trie for the second string but using the trie
    that has been created by the first string. Since the purpose is only to find the longest
    common string, once it cannot find a similar child in the trie, it will just skip. So there
    is no additional nodes added to the trie.
    :Input:
        :param trie: The suffix trie created by the first input string of the main function
        :param suffix: The list of suffix outputted by the second string
        :param max_string: The maximum common string that can be found from the 2 strings
    :return:
    :Time complexity: O(N^2+M^2) where N or M is the smaller string. This is because unlike
    the insertion_suffix, it only reaches the point where they do not see any further child
    and then stop.
    :Aux space complexity: O(N^2) taking from the insertion_suffix since in this function,
    no new node is added. Only traversing the 2nd string suffixes together with checking.
    """
    # UP TO SOME LINES ARE SIMILAR TO INSERTION_SUFFIX SO IT WON'T BE EXPLAINED
    node = trie.nodes[0]
    for char in suffix:
        node_changed = False
        check_exist = False

        for child in range(len(node.children)):
            if node.children[child].individual_char == char:
                node = node.children[child]
                node_changed = True
                break
            if child == len(node.children) - 1:
                check_exist = True

        if check_exist is True:
            break

        # check if there is a change of node (where the current node becomes the children node since it found
        # the same character
        if node_changed is True:
            if len(max_string) < len(node.chars_collected):      # returns the longer list (list of chars)
                max_string = node.chars_collected
    return max_string


def fix_math(num):
    """
    A very simple method which simply just round off numbers in a more exact
    manner since round() rounds down 0.5 when it should round up.
    :Input:
        :param num: Any numbers to be rounded off
    :return: an integer after rounding off
    :Time complexity: O(1) by computation of integer
    :Aux space complexity: O(1) no list is created
    """
    if num < math.floor(num) + 0.5:     # indicating that num has not reaches +0.5
        return math.floor(num)
    else:
        return math.ceil(num)


def compare_subs(submission1, submission2):
    """
    This method just calls methods that was used to create suffix trie and the
    comparison. It mainly count the percentage of the similar string of the
    2 strings and return the strings and percentages,
    :Input:
        :param submission1: The first input string
        :param submission2: The second input string
    :return: [the longest string, percentage of input1, percentage of input2]
    :Time complexity: O(N^2+M^2) where a suffix trie is created. This contains
    the same explanation as the methods above since the biggest complexity is
    obtained from calling the function
    :Aux space complexity: O(N^2) where N is the first string and the suffix
    trie created is only for the first string since in the second string, the
    trie is not updated
    """
    sub1 = "".join([submission1, "$"])      # TC: O(N) where N is the length of string 1
    sub2 = "".join([submission2, "#"])      # TC: O(M) where M is the length of string 2
    suffix_trie = SuffixTrie(collect_suffix(sub1))
    string2suffixes = collect_suffix(sub2)
    max_string = []     # ASC: O(n) where worst is min(len(submission1), len(submission2))
    # traversing through the suffix trie that has been created
    for suffix in string2suffixes:
        max_string = second_sentence(suffix_trie, suffix, max_string)
    num1 = 0
    num2 = 0

    # concatenate the list to a single string
    strings = "".join(max_string)

    # condition to avoid zero divisor
    if len(submission1) > 0:
        num1 = len(strings) * 100 / len(submission1)
    if len(submission2) > 0:
        num2 = len(strings) * 100 / len(submission2)

    # getting the percentage
    percent1 = fix_math(num1)
    percent2 = fix_math(num2)

    return [strings, percent1, percent2]

#
# if __name__ == "__main__":
#     # s1 = "radix sort and counting sort are both non comparison sorting algorithms"
#     # s2 = "counting sort and radix sort are both non comparison sorting algorithms"
#     # print(compare_subs(s1, s2))
#
#     available = [[2, 0, 2, 1, 2], [3, 3, 1, 0, 0], [0, 1, 0, 3, 0], [0, 0, 2, 0, 3], [1, 0, 0, 2, 1], [0, 0, 3, 0, 2],
#                  [0, 2, 0, 1, 0], [1, 3, 3, 2, 0], [0, 0, 1, 2, 1], [2, 0, 0, 3, 0]]
#
#     print(allocate(available))
