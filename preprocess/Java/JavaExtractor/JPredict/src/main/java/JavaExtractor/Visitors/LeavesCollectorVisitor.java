package JavaExtractor.Visitors;

import JavaExtractor.Common.Common;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.visitor.TreeVisitor;

import java.util.ArrayList;
import java.util.List;

public class LeavesCollectorVisitor extends TreeVisitor {
    private ArrayList<ASTNode> astNodes = new ArrayList<>();

    @Override
    public void process(Node node) {
        if (node instanceof Comment)
            return;
        String type=node.getClass().getSimpleName();
        int id=node.getUserData(Common.id);
        int parent=-1;
        ArrayList<Integer> children=new ArrayList<>();
        if(!node.getParentNode().getClass().getSimpleName().toString().equals("CompilationUnit"))
            parent=node.getParentNode().getUserData(Common.id);
        for(Node n:node.getChildrenNodes())
            children.add(n.getUserData(Common.id));
        int begin_row=node.getRange().begin.line;
        int begin_col=node.getRange().begin.column;
        int end_row=node.getRange().end.line;
        int end_col=node.getRange().end.column;

        ASTNode astNode=new ASTNode(id,parent,children,type,begin_row,begin_col,end_row,end_col);
        astNodes.add(astNode);

//        System.out.println(node);
//        System.out.println(id);
//        System.out.println(parent);
//        System.out.println(children);
//        System.out.println(type);
//        System.out.println(node.getRange());
//        System.out.println(begin_row+","+begin_col+" "+end_row+","+end_col);
//        System.out.println("__________________________________");
    }

    public ArrayList<ASTNode> getAstNodes() {
        return astNodes;
    }
}
