package JavaExtractor;

import java.util.ArrayList;
import java.util.stream.Collectors;
import java.util.stream.Stream;

public class Subtokenizer {

    public static final String EmptyString = "";
    public static final String BlankWord = "BLANK";
    public static final String internalSeparator = "|";
    public static String subtokenize(String original) {
        String normalizedMethodName = normalizeName(original, BlankWord);
        ArrayList<String> splitNameParts = splitToSubtokens(original);
        String splitName = normalizedMethodName;
        if (splitNameParts.size() > 0)
            splitName = String.join(internalSeparator, splitNameParts);
        return splitName;
    }

    public static String normalizeName(String original, String defaultString) {
        original = original.replaceAll("\\\\n", "") // escaped new lines
                .replaceAll("\\s+", "") // whitespaces
                .replaceAll("[\"']", "") // quotes, apostrophies, commas
                .replaceAll("\\P{Print}", ""); // unicode weird characters
        //String stripped = original.replaceAll("[^A-Za-z]", "");
        String stripped = original;
        if (stripped.length() == 0) {
            String carefulStripped = original.replaceAll(" ", "_");
            if (carefulStripped.length() == 0) 
                return defaultString;
            else 
                return carefulStripped;
        } else 
            return stripped;

    }

    private static ArrayList<String> splitToSubtokens(String str1) {
        String str2 = str1.replace("|", " ");
        String str3 = str2.trim();
        for (String retval: str3.split("(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\\s+")){
            System.out.println(retval);
        }
        return Stream.of(str3.split("(?<=[a-z])(?=[A-Z])|_|[0-9]|(?<=[A-Z])(?=[A-Z][a-z])|\\s+"))
                .filter(s -> s.length() > 0).map(s -> normalizeName(s, EmptyString))
                .filter(s -> s.length() > 0).collect(Collectors.toCollection(ArrayList::new));
    }

}
