from enum import Enum, unique


@unique
class CompilerAction(Enum):
    AppendFirst = 1
    AppendLast = 2
    RemoveFirst = 3
    RemoveLast = 4

    @staticmethod
    def is_append(
        value: int
    ) -> bool:
        return (value == CompilerAction.AppendFirst
                or value == CompilerAction.AppendLast)

    @staticmethod
    def is_remove(
        value: int
    ) -> bool:
        return (value == CompilerAction.RemoveFirst
                or value == CompilerAction.RemoveLast)
