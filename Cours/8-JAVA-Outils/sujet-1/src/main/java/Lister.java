import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.stream.IntStream;

public class Lister {
//
//    public static void lister (int number) {
//        if (number % 3 ==0 && number % 5 != 0) {
//            System.out.println("Fizz");
//        } else if (number % 5 == 0 && number % 3 != 0) {
//            System.out.println("Buzz");
//        } else if (number % 3 == 0) {
//            System.out.println("FizzBuzz");
//        } else {
//            System.out.println(number);
//        }
//    }
//
//    public static void main(String[] args) {
//        for (int i = 0; i <= 10000; i++ ) {
//            lister(i);
//        }
//    }





//public static void lister(int number) {
//    // Stockez temporairement le résultat au lieu de l'afficher immédiatement
//    String result;
//    if (number % 3 == 0 && number % 5 != 0) {
//        result = "Fizz";
//    } else if (number % 5 == 0 && number % 3 != 0) {
//        result = "Buzz";
//    } else if (number % 3 == 0) {
//        result = "FizzBuzz";
//    } else {
//        result = Integer.toString(number);
//    }
//    // Affichez le résultat
//    System.out.println(result);
//}
//
//    public static void main(String[] args) {
//        int numberOfThreads = Runtime.getRuntime().availableProcessors();
//        ExecutorService executor = Executors.newFixedThreadPool(numberOfThreads);
//
//
//        CompletableFuture<?>[] futures = IntStream.rangeClosed(0, 1000000)
//                .mapToObj(i -> CompletableFuture.runAsync(() -> lister(i), executor))
//                .toArray(CompletableFuture[]::new);
//
//
//        CompletableFuture.allOf(futures).join();
//
//        executor.shutdown();
//    }

    public static CompletableFuture<String> verifierFizz(int number) {
        return CompletableFuture.supplyAsync(() -> (number % 3 == 0) ? "Fizz" : "");
    }

    public static CompletableFuture<String> verifierBuzz(int number) {
        return CompletableFuture.supplyAsync(() -> (number % 5 == 0) ? "Buzz" : "");
    }

    public static CompletableFuture<String> verifierFizzBuzz(int number) {
        return CompletableFuture.supplyAsync(() -> (number % 3 == 0 && number % 5 == 0) ? "FizzBuzz" : "");
    }

    public static void lister(int number) {
        // Exécuter les vérifications en parallèle et combiner les résultats
        CompletableFuture<String> fizzFuture = verifierFizz(number);
        CompletableFuture<String> buzzFuture = verifierBuzz(number);
        CompletableFuture<String> fizzBuzzFuture = verifierFizzBuzz(number);

        CompletableFuture<Void> resultatFinal = CompletableFuture.allOf(fizzFuture, buzzFuture, fizzBuzzFuture)
                .thenRun(() -> {
                    try {
                        String fizz = fizzFuture.get();
                        String buzz = buzzFuture.get();
                        String fizzBuzz = fizzBuzzFuture.get();

                        if (!fizzBuzz.isEmpty()) {
                            System.out.println("FizzBuzz");
                        } else if (!fizz.isEmpty() && buzz.isEmpty()) {
                            System.out.println("Fizz");
                        } else if (!buzz.isEmpty() && fizz.isEmpty()) {
                            System.out.println("Buzz");
                        } else {
                            System.out.println(number);
                        }
                    } catch (Exception e) {
                        e.printStackTrace();
                    }
                });

        // Assurer que le résultat final est calculé avant de continuer
        resultatFinal.join();
    }

    public static void main(String[] args) {
        ExecutorService executor = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors());

        // Exécuter l'analyse pour chaque nombre en parallèle
        IntStream.rangeClosed(0, 1000000).forEach(i ->
                CompletableFuture.runAsync(() -> lister(i), executor)
        );
        executor.shutdown();

        int tab[] = new int [8];
    }

}
