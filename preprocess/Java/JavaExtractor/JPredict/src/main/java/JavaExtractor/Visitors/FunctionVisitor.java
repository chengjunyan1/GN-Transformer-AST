package JavaExtractor.Visitors;

import JavaExtractor.Common.CommandLineValues;
import com.github.javaparser.ast.body.ClassOrInterfaceDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.util.ArrayList;

@SuppressWarnings("StringEquality")
public class FunctionVisitor extends VoidVisitorAdapter<Object> {
    private final CommandLineValues m_CommandLineValues;
    private ArrayList<ASTNode> astNodes = new ArrayList<>();

    public FunctionVisitor(CommandLineValues commandLineValues) {
        this.m_CommandLineValues = commandLineValues;
    }

    @Override
    public void visit(ClassOrInterfaceDeclaration node, Object arg) {
        visitMethod(node);

        super.visit(node, arg);
    }

    private void visitMethod(ClassOrInterfaceDeclaration node) {
        NodeInit nodeInit=new NodeInit();
        nodeInit.visitDepthFirst(node);
        LeavesCollectorVisitor leavesCollectorVisitor = new LeavesCollectorVisitor();
        leavesCollectorVisitor.visitDepthFirst(node);
        astNodes = leavesCollectorVisitor.getAstNodes();
    }

    public ArrayList<ASTNode> getAstNodes() {
        return astNodes;
    }
}
