package JavaExtractor.Visitors;

import JavaExtractor.Common.Common;
import JavaExtractor.FeaturesEntities.Property;
import com.github.javaparser.ast.Node;
import com.github.javaparser.ast.UserDataKey;
import com.github.javaparser.ast.comments.Comment;
import com.github.javaparser.ast.expr.NullLiteralExpr;
import com.github.javaparser.ast.stmt.Statement;
import com.github.javaparser.ast.type.ClassOrInterfaceType;
import com.github.javaparser.ast.visitor.TreeVisitor;

import java.util.ArrayList;
import java.util.List;

public class NodeInit extends TreeVisitor {

    private int index = 0;
    @Override
    public void process(Node node) {
        if (node instanceof Comment)
            return;
        node.setUserData(Common.id,index);
        this.index+=1;
    }

}
