package JavaExtractor;

import JavaExtractor.Common.CommandLineValues;
import JavaExtractor.Common.Common;
import JavaExtractor.Visitors.FunctionVisitor;
import JavaExtractor.Visitors.ASTNode;
import com.alibaba.fastjson.JSONArray;
import com.alibaba.fastjson.JSONObject;
import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseProblemException;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;

import java.util.ArrayList;

@SuppressWarnings("StringEquality")
class FeatureExtractor {
    private final CommandLineValues m_CommandLineValues;
    private String src;

    public FeatureExtractor(CommandLineValues commandLineValues) {
        this.m_CommandLineValues = commandLineValues;
    }

    public JSONArray extractAST(String code,  String fid, JSONArray jsonArray) {
        CompilationUnit m_CompilationUnit = parseFileWithRetries(code);

        if(m_CompilationUnit==null) return jsonArray;
        FunctionVisitor functionVisitor = new FunctionVisitor(m_CommandLineValues);
        functionVisitor.visit(m_CompilationUnit, null);
        ArrayList<ASTNode> ast = functionVisitor.getAstNodes();

        JSONObject jsonObject=new JSONObject();
        jsonObject.put("fid", fid);
        jsonObject.put("src", src);
        jsonObject.put("ast", ast);
        jsonArray.add(jsonObject);
        return jsonArray;
    }

    private CompilationUnit parseFileWithRetries(String code) {
        final String classPrefix = "public class UnknownClass {\n";
        final String classSuffix = "\n}";

        String content = code;
        CompilationUnit parsed;
        try {
            parsed = JavaParser.parse(content);
        } catch (ParseProblemException e1) {
            try {
                content = classPrefix + code + classSuffix;
                parsed = JavaParser.parse(content);
            } catch (ParseProblemException e2) {
                return null;
            }
        }
        src=content;
        return parsed;
    }

}
