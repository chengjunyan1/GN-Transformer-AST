PAD = 0
UNK = 1
BOS = 2
EOS = 3

PAD_WORD = '<blank>'
UNK_WORD = '<unk>'
BOS_WORD = '<s>'
EOS_WORD = '</s>'

DATA_LANG_MAP = {
    'java': 'java',
    'python': 'python'
}

LANG_ID_MAP = {
    'java': 0,
    'python': 1,
    'c#': 2
}

# WE DO NOT NEED THE FOLLOWING VOCABS IN INPUTTER, ITS JUST FOR SEE, BUT THE MODEL WILL READ THE SIZE OF THEM

WORD_TYPE_VOCAB={
    '<PAD>': 0,
    '<UNK>': 1,
    'Modifier': 2,
    'Keyword': 3,
    'Identifier': 4,
    'Separator': 5,
    'Annotation': 6,
    'BasicType': 7,
    'Operator': 8,
    'Null': 9,
    'DecimalInteger': 10,
    'Boolean': 11,
    'String': 12,
    'HexInteger': 13,
    'DecimalFloatingPoint': 14,
    'OctalInteger': 15,
    'BinaryInteger': 16
 }

NODE_TYPE_VOCAB={
    '<PAD>': 0,
    '<UNK>': 1,
    'null': 2,
    'ClassOrInterfaceDeclaration': 3,
    'NameExpr': 4,
    'MethodDeclaration': 5,
    'MarkerAnnotationExpr': 6,
    'PrimitiveType': 7,
    'Parameter': 8,
    'VariableDeclaratorId': 9,
    'ArrayBracketPair': 10,
    'ClassOrInterfaceType': 11,
    'BlockStmt': 12,
    'TryStmt': 13,
    'IfStmt': 14,
    'BinaryExpr': 15,
    'NullLiteralExpr': 16,
    'ReturnStmt': 17,
    'UnaryExpr': 18,
    'IntegerLiteralExpr': 19,
    'ExpressionStmt': 20,
    'AssignExpr': 21,
    'MethodCallExpr': 22,
    'VariableDeclarationExpr': 23,
    'VariableDeclarator': 24,
    'CatchClause': 25,
    'ThrowStmt': 26,
    'FieldAccessExpr': 27,
    'ArrayAccessExpr': 28,
    'WhileStmt': 29,
    'ObjectCreationExpr': 30,
    'ThisExpr': 31,
    'VoidType': 32,
    'BooleanLiteralExpr': 33,
    'StringLiteralExpr': 34,
    'ConstructorDeclaration': 35,
    'ExplicitConstructorInvocationStmt': 36,
    'CastExpr': 37,
    'ClassExpr': 38,
    'WildcardType': 39,
    'ArrayCreationExpr': 40,
    'ArrayCreationLevel': 41,
    'ForStmt': 42,
    'ConditionalExpr': 43,
    'EnclosedExpr': 44,
    'CharLiteralExpr': 45,
    'SynchronizedStmt': 46,
    'InstanceOfExpr': 47,
    'ForeachStmt': 48,
    'SingleMemberAnnotationExpr': 49,
    'DoubleLiteralExpr': 50,
    'SwitchStmt': 51,
    'SwitchEntryStmt': 52,
    'BreakStmt': 53,
    'LabeledStmt': 54,
    'ContinueStmt': 55,
    'UnionType': 56,
    'ArrayInitializerExpr': 57,
    'SuperExpr': 58,
    'EmptyStmt': 59,
    'TypeParameter': 60,
    'FieldDeclaration': 61,
    'LongLiteralExpr': 62,
    'AssertStmt': 63,
    'ArrayType': 64,
    'NormalAnnotationExpr': 65,
    'MemberValuePair': 66,
    'DoStmt': 67,
    'QualifiedNameExpr': 68,
    'InitializerDeclaration': 69,
    'LongLiteralMinValueExpr': 70,
    'IntegerLiteralMinValueExpr': 71
}