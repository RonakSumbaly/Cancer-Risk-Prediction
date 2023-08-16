class Solution {
    public int minFallingPathSum(int[][] matrix) {
        int n = matrix.length; 
        if(n == 1) return matrix[0][0];
        
        int[] prev = new int[n];
        int[] curr = new int[n];

        for(int i = 0; i < n; i++)
            prev[i] = matrix[0][i];

        int answer = Integer.MAX_VALUE;
        
        for(int i = 1; i < n; i++) {
            for(int j = 0 ; j < n; j ++) {
                int up = matrix[i][j] + prev[j];
                int topLeft = j - 1 >= 0 ? matrix[i][j] + prev[j-1] : (int) 1e9;
                int topRight = j + 1 < n ? matrix[i][j] + prev[j+1] : (int) 1e9;
                curr[j] = Math.min(up, Math.min(topLeft, topRight));
            }
            prev = curr.clone();
        }

        for(int i = 0; i < n; i++) {
            answer = Math.min(answer, prev[i]);
        }

        return answer; 
    }

    public int minFallPath(int[][] matrix, Integer[][] memo, int n, int row, int index) {
        if(row >= n || index < 0 || index >= n) return (int) 1e9;
        if(row == n - 1) return matrix[row][index];
        if(memo[row][index] != null) return memo[row][index];

        memo[row][index] = matrix[row][index] + Math.min(minFallPath(matrix, memo, n, row+1, index), Math.min(minFallPath(matrix, memo, n,  row+1, index+1), minFallPath(matrix, memo, n, row+1, index-1)));        

        return memo[row][index];
    }
}
