function getData(){
    
}

window.addEventListener("DOMContentLoaded", (event) => {
    const log = document.getElementById("data")
    const count = document.getElementById("count0")
    if (log) {
        log.addEventListener("input", updateValue)
        function updateValue(e){
            let temp = e.target.value
            // temp = temp.toString().replace(/(?!\s+$)\s+/g, ",")
            // log.value = temp
            let arr = log.value.split('\n')
            arr = arr.filter(a => a != '')
            count.innerHTML = arr.length
            
        }
    }
})