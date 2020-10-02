package JavaExtractor.Visitors;

import java.util.ArrayList;

public class ASTNode {

    private int id;
    private int parent;
    private ArrayList<Integer> children;
    private String type;
    private int begin_row;
    private int begin_col;
    private int end_row;
    private int end_col;

    public ASTNode(int id, int parent, ArrayList<Integer> children, String type, int begin_row, int begin_col, int end_row, int end_col) {
        super();
        this.id=id;
        this.parent=parent;
        this.children=children;
        this.type=type;
        this.begin_row=begin_row;
        this.begin_col=begin_col;
        this.end_row=end_row;
        this.end_col=end_col;
    }

    public int getId() {
        return id;
    }

    public int getParent() {
        return parent;
    }

    public String getType() {
        return type;
    }

    public ArrayList<Integer> getChildren() {
        return children;
    }

    public int getBegin_row() {
        return begin_row;
    }

    public int getBegin_col() {
        return begin_col;
    }

    public int getEnd_row() {
        return end_row;
    }

    public int getEnd_col() {
        return end_col;
    }
}
