'''
Defines the CompilerAction enumeration.
'''
from enum import Enum, unique


@unique
class CompilerAction(Enum):
    '''
    Defines the possible actions taken at each step by the STOQ compiler.
    '''
    AppendFirst = 1
    AppendLast = 2
    RemoveFirst = 3
    RemoveLast = 4

    @staticmethod
    def is_append(
        value: int
    ) -> bool:
        '''
        Returns whether the given CompilerAction enumeration value
        is an append action.

        :param value: The CompilerAction enumeration value.
        :type value: int
        :return: Whether the value is an append action.
        :rtype: bool
        '''
        return (value == CompilerAction.AppendFirst
                or value == CompilerAction.AppendLast)

    @staticmethod
    def is_remove(
        value: int
    ) -> bool:
        '''
        Returns whether the given CompilerAction enumeration value
        is a remove action.

        :param value: The CompilerAction enumeration value.
        :type value: int
        :return: Whether the value is a remove action.
        :rtype: bool
        '''
        return (value == CompilerAction.RemoveFirst
                or value == CompilerAction.RemoveLast)
