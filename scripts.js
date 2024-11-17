 $(document).ready(function() {
  
        $("#vidlink li").css("display", "none"), $("#vidlink .Sports").css("display", ""), $(".channel-list .nav-link").click(function() {
            $(".channel-list .nav-link").removeClass("active"), $(this).addClass("active");
            var i = $(this).data("type");
            $("#vidlink li").css("display", "none"), $("#vidlink ." + i).css("display", "")
        }), $(".channel").click(function() {
            vsgLoadVideo($(this).data("link"))
        })
});