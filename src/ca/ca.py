from copy import deepcopy
from typing import Tuple, Union, List


class CA(object):
    """
    A class to represent a 1D cellular automaton.

    ...

    Attributes
    -------
    rule : List[int]
        list of neighborhood to state transitions
    nhd : Tuple[int, int]
        tuple representing the left and right sizes of the neighborhood
    n_states : int
        the number of states the automaton can operate on

    Methods
    -------
    __call__(seq: List[int], steps: int, pad: bool = True) -> List[List[int]]:
        Applies the cellular automaton <steps> times, using <seq> as the
        initial set of states. Returns the full state transition history of
        the automaton. If <pad> is True, all lists of states in the output
        will be padded to the same length.
    """
    def __init__(
        self,
        rule: Union[List[int], int],
        nhd: Tuple[int, int] = (1, 1),
        n_states: int = 2,
    ) -> None:
        """
        Constructs the necessary attributes to represent a 1D cellular automaton.

        Args:
            rule (Union[List[int], int]): The cellular automaton rule, represented
                as either a list of state transitions or as in integer using
                Wolfram numbering.
            nhd (Tuple[int, int], optional): The left and right neighborhood sizes. 
                Defaults to (1, 1).
            n_states (int, optional): The number of states used by the automaton. 
                Defaults to 2.

        Raises:
            ValueError: Invalid neighborhood specification.
            ValueError: Invalid number of states.
            ValueError: Invalid rule number.
            ValueError: Incorrect rule list length.
            ValueError: Invalid state in rule definition.
        """
        if nhd[0] < 0 or nhd[1] < 0:
            raise ValueError('neighborhood sizes must be at least 0')
        
        if n_states < 2:
            raise ValueError('number of states must be at least 2')
        
        self.nhd = nhd
        self.n_states = n_states
        # neighborhood size is <size left> + 1 + <size_right>
        self.in_size = sum(nhd) + 1
        
        # maximum Wolfram number for a CA with <n_states> states and input size <self.in_size>
        max_rule_num = n_states ** (n_states ** self.in_size) - 1
        rule_len = n_states ** self.in_size
        
        # check for invalid rule specification
        if isinstance(rule, int):
            # case: rule is a Wolfram number
            if rule < 0 or rule > max_rule_num:
                raise ValueError(
                    f'rule number with {n_states} and neighborhood size {self.in_size} must be between 0 and {max_rule_num}'
                )
            self.rule = self._int_to_states(rule)
        else:
            # case: rule is a list of state transitions
            if len(rule) != rule_len:
                raise ValueError(
                    f'rule list with {n_states} and neighborhood size {self.in_size} must have length {rule_len}'
                )
            if not all(map(lambda x: 0 <= x < n_states, rule)):
                raise ValueError(
                    f'rule list with {n_states} must specify state between 0 and {n_states - 1} for each element'
                )
            self.rule = rule
            
        self._max_rule_ix = rule_len
        
    def __call__(
        self, seq: List[int], steps: int = 1, pre_pad: bool = True, post_pad = True,
    ) -> List[List[int]]:
        """Apply cellular automaton rule to an input for a specified number
        of steps.

        Args:
            seq (List[int]): Initial state sequence.
            steps (int, optional): Number of steps to apply the automaton. 
                Defaults to 1.
            pre_pad (bool, optional): If True, make each sequence have fixed length
                by padding zeros to inputs before applying the automaton rule. 
                Defaults to True.
            post_pad (bool, optional): If True, make each sequence have fixed length
                by padding zeros after applying the automaton rule. 
                Defaults to True.

        Raises:
            ValueError: Non-positive number of steps.

        Returns:
            List[List[int]]: List of state sequences, representing the evolution
                of the state system throughout the history of the automaton.
        """
        if steps < 1:
            raise ValueError(
                f'number of steps to process must be at least 1'
            )
            
        # begin with the initial state sequence
        # at each step, apply the CA rule to the most recent state sequence
        
        out = [deepcopy(seq)]
        for _ in range(steps):
            states_next = self._apply_rule(out[-1], pre_pad)
            if len(states_next) > 0:
                out.append(states_next)
            else:
                break
            
        if post_pad:
            out = self._pad_output(out)
            
        return out

    def _apply_rule(self, seq: List[int], pre_pad: bool = True) -> List[int]:
        """Applies the cellular automaton for a single step

        Args:
            seq (List[int]): State sequence.
            pre_pad (bool, optional): Pad input with zeros to produce outputs with same 
                length as the original input. Defaults to True.

        Raises:
            ValueError: Sequence length too short.
            ValueError: State in sequence out of bounds.

        Returns:
            List[int]: State sequence after applying automaton rule.
        """
        # check input length
        if not pre_pad and len(seq) < self.in_size:
            raise ValueError(
                f'input sequences without padding must have length at least the size of the neighborhood'
            )
        
        # check for out of bounds states
        if not all(map(lambda x: 0 <= x < self.n_states, seq)):
            raise ValueError(
                f'input sequence must be integers between 0 and {self.n_states - 1}'
            )
            
        out = []
        
        # if pad, prepend and append zeroes to input to produce output with the same length
        # as <seq>
        if pre_pad:
            seq_ = (
                [0 for _ in range(self.nhd[0])] + seq + [0 for _ in range(self.nhd[1])]
            )
        else:
            seq_ = seq
            
        # for a sliding window of length <self.in_size>, apply the cellular automaton to each
        # window; append the output of the automaton to method output
        for start_ix in range(len(seq_) - self.in_size + 1):
            window = seq_[start_ix : start_ix + self.in_size]
            window_int = self._neighborhood_to_index(window)
            out.append(self.rule[window_int])
            
        return out

    def _pad_output(self, seqs: List[List[int]]) -> List[List[int]]:
        """_summary_

        Args:
            seqs (List[List[int]]): List of state sequences to pad.

        Returns:
            List[List[int]]: Lost of state sequences zero-padded to equal length.
        """
        
        # run checks on input sequences
        for seq in seqs:
            self._states_in_bounds(seq)
        
        # if <self.in_size> is 1, then applying the cellular automaton preserves input length
        if self.in_size > 1:
            # determine the length of the longest sequence in the list -- under normal circumstances
            # this should be the first member of the list
            seq_lens = list(map(len, seqs))
            max_len = max(seq_lens)
            min_len = min(seq_lens)
            # if <max_len> is equal to <min_len> there's nothing to do
            if max_len > min_len:
                # pad zeros to the current sequence until it has the same length as the longest sequence
                for ix, seq_len in enumerate(seq_lens):
                    fill = max_len - seq_len
                    if fill > 0:
                        # the left and right neighborhood sizes may be different so we determine how many
                        # repetitions of both neighborhoods to pad; if there's a remainder, we arbitrarily
                        # add the extra padding to the left side
                        nhd_reps, extra = divmod(fill, self.in_size - 1)
                        extra_both, extra_l = divmod(extra, 2)
                        
                        l_pad = [0 for _ in range(nhd_reps * self.nhd[0] + extra_both + extra_l)]
                        r_pad = [0 for _ in range(nhd_reps * self.nhd[1] + extra_both)]
                        
                        seqs[ix] = l_pad + seqs[ix] + r_pad
                    
        return seqs

    def _neighborhood_to_index(self, states: List[int]) -> int:
        """Converts a neighborhood state sequence into an integer index for lookup in the rule list.

        Args:
            states (List[int]): State sequence.

        Returns:
            int: Index of the corresponding state sequence in the rule list.
        """
        self._states_in_bounds(states)
        return sum([state * (self.n_states ** (len(states) - p - 1)) for (p, state) in enumerate(states)])

    def _int_to_states(self, k: int) -> List[int]:
        """Converts an integer index from the rule list into a sequence of states. Can
        also be used to convert a Wolfram number for a cellular automaton into a rule
        list.

        Args:
            k (int): _description_

        Returns:
            List[int]: _description_
        """
        
        # recursively compute the state sequence in little endian format
        # by successively taking modulus and division by <self.n_states>
        def int_to_states_rcrs(k: int, n_states: int, acc: List[int]) -> List[int]:
            if k == 0:
                return acc
            else:
                next, state = divmod(k, n_states)
                acc.append(state)
                return int_to_states_rcrs(next, n_states, acc)

        out = []
        out = int_to_states_rcrs(k, self.n_states, out)
        
        rem = (self.n_states ** self.in_size) - len(out)
        if rem > 0:
            out += [0 for _ in range(rem)]
        
        return out
    
    def _states_in_bounds(self, states: List[int]) -> None:
        """Checks that all states in a sequence are within bounds (i.e., between 0 and <self.n_states> - 1) 

        Args:
            states (List[int]): State sequence.

        Raises:
            ValueError: State out of bounds.
        """
        for state in states:
            if state < 0 or state >= self.n_states:
                raise ValueError(f'all states must be between 0 and {self.n_states - 1}')

    def _check_rule_ix(self, rule_ix: int) -> None:
        """Checks 

        Args:
            state_num (int): _description_

        Raises:
            ValueError: _description_
        """
        if rule_ix < 0 or rule_ix > self._max_rule_ix:
            raise ValueError(f'integer representation of states must be between 0 and {self._max_rule_ix}')
